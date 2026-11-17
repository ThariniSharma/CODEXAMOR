import { GoogleGenAI } from '@google/genai';
import JSZip from 'jszip';
import mammoth from 'mammoth';
import { extractText } from 'unpdf';
import * as XLSX from 'xlsx';

export interface ParsedUploadResult {
  title: string;
  fileType: string;
  content: string;
}

const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif']);

function stripXmlTags(xml: string): string {
  return xml
    .replace(/<a:br\s*\/?>/gi, '\n')
    .replace(/<\/a:p>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractPrintableLegacyText(buffer: Buffer): string {
  const utf8 = buffer.toString('utf8');
  const runs = utf8.match(/[A-Za-z0-9][A-Za-z0-9 ,.;:!?'"()\-/%]{8,}/g) ?? [];
  const filtered = runs.filter((chunk) => !/^(?:Root Entry|WordDocument|PowerPoint Document|SummaryInformation)/.test(chunk));
  return filtered.join('\n').slice(0, 20000).trim();
}

async function extractImageDescription(
  buffer: Buffer,
  mimeType: string,
  filename: string,
  notes?: string,
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (apiKey) {
    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
      });
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: mimeType || 'image/png',
                data: buffer.toString('base64'),
              },
            },
            {
              text: 'Extract any visible text (OCR) and summarize the key details in this image clearly for document search.',
            },
          ],
        },
      });
      const extracted = response.text?.trim();
      if (extracted) {
        return notes ? `${extracted}\n\nUser Notes: ${notes}` : extracted;
      }
    } catch {
      // Fallback to metadata + user notes if vision API is rate-limited or unavailable
    }
  }

  const sizeKb = Math.max(1, Math.round(buffer.length / 1024));
  const cleanName = filename.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
  const parts = [`Image Document: ${filename} (${cleanName}, ${mimeType || 'image'}, ${sizeKb} KB)`];
  if (notes?.trim()) {
    parts.push(`Notes: ${notes.trim()}`);
  }
  return parts.join('\n');
}

export async function parseUploadedDocument(params: {
  filename: string;
  mimeType?: string;
  base64Data: string;
  title?: string;
  notes?: string;
}): Promise<ParsedUploadResult> {
  const { filename, mimeType = '', base64Data, title, notes } = params;
  const ext = filename.split('.').pop()?.toLowerCase() ?? 'txt';
  const cleanTitle = (title?.trim() || filename.replace(/\.[^.]+$/, '') || 'Uploaded Document').slice(0, 120);
  const buffer = Buffer.from(base64Data, 'base64');

  if (buffer.length === 0) {
    throw new Error('The uploaded file appears to be empty.');
  }

  if (ext === 'pdf' || mimeType === 'application/pdf') {
    const result = await extractText(new Uint8Array(buffer), { mergePages: true });
    const text = (Array.isArray(result.text) ? result.text.join('\n') : String(result.text || '')).trim();
    if (!text) {
      throw new Error('Could not find selectable text inside this PDF. Try a text-based PDF or upload screenshots as images.');
    }
    return {
      title: cleanTitle,
      fileType: 'PDF',
      content: text.slice(0, 25000),
    };
  }

  if (ext === 'docx' || mimeType.includes('wordprocessingml')) {
    const { value } = await mammoth.extractRawText({ buffer });
    const text = value.trim();
    if (!text) {
      throw new Error('This Word (.docx) document has no readable text.');
    }
    return {
      title: cleanTitle,
      fileType: 'Word',
      content: text.slice(0, 25000),
    };
  }

  if (ext === 'doc') {
    const text = extractPrintableLegacyText(buffer);
    if (!text) {
      throw new Error('Legacy binary .doc could not be parsed. Please save as .docx or .pdf and upload again.');
    }
    return { title: cleanTitle, fileType: 'Word', content: text };
  }

  if (ext === 'pptx' || mimeType.includes('presentationml')) {
    const zip = await JSZip.loadAsync(buffer);
    const slideFiles = Object.keys(zip.files)
      .filter((name) => /^ppt\/slides\/slide\d+\.xml$/i.test(name))
      .sort((a, b) => {
        const na = Number(a.match(/slide(\d+)\.xml/)?.[1] ?? 0);
        const nb = Number(b.match(/slide(\d+)\.xml/)?.[1] ?? 0);
        return na - nb;
      });

    const slides: string[] = [];
    for (const file of slideFiles) {
      const xml = await zip.files[file].async('string');
      const slideText = stripXmlTags(xml);
      if (slideText) {
        const num = file.match(/slide(\d+)\.xml/)?.[1] ?? '';
        slides.push(`Slide ${num}: ${slideText}`);
      }
    }
    const combined = slides.join('\n\n').trim();
    if (!combined) {
      throw new Error('No text found in PowerPoint slides.');
    }
    return {
      title: cleanTitle,
      fileType: 'PowerPoint',
      content: combined.slice(0, 25000),
    };
  }

  if (ext === 'ppt') {
    const text = extractPrintableLegacyText(buffer);
    if (!text) {
      throw new Error('Legacy binary .ppt could not be parsed. Please export as .pptx or .pdf and upload again.');
    }
    return { title: cleanTitle, fileType: 'PowerPoint', content: text };
  }

  if (ext === 'xlsx' || ext === 'xls' || ext === 'csv' || mimeType.includes('spreadsheet') || mimeType.includes('excel')) {
    const workbook = XLSX.read(buffer, { type: 'buffer' });
    const sheetTexts: string[] = [];
    for (const name of workbook.SheetNames) {
      const sheet = workbook.Sheets[name];
      const csv = XLSX.utils.sheet_to_csv(sheet, { blankrows: false }).trim();
      if (csv) {
        sheetTexts.push(`Sheet: ${name}\n${csv}`);
      }
    }
    const combined = sheetTexts.join('\n\n').trim();
    if (!combined) {
      throw new Error('The spreadsheet contains no readable rows.');
    }
    return {
      title: cleanTitle,
      fileType: 'Excel',
      content: combined.slice(0, 25000),
    };
  }

  if (IMAGE_EXTENSIONS.has(ext) || mimeType.startsWith('image/')) {
    const content = await extractImageDescription(buffer, mimeType || `image/${ext}`, filename, notes);
    return {
      title: cleanTitle,
      fileType: 'Image',
      content: content.slice(0, 12000),
    };
  }

  // Plain text / Markdown / JSON / Fallback
  const text = buffer.toString('utf8').trim();
  if (!text) {
    throw new Error('No readable text found in the uploaded file.');
  }
  return {
    title: cleanTitle,
    fileType: ext === 'md' ? 'Markdown' : ext === 'json' ? 'JSON' : 'Text',
    content: text.slice(0, 25000),
  };
}

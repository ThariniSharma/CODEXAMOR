import { useMemo, useState, type ChangeEvent, type FormEvent } from 'react';
import { ChevronDown, ChevronUp, FileText, Lock, Plus, Search, Trash2, Upload } from 'lucide-react';
import { rankLocalRagDocuments } from '../lib/tfidfRag';
import type { LocalRagDocument } from '../lib/types';

interface PrivacyRagViewProps {
  docs: LocalRagDocument[];
  customTerms: string[];
  onChangeCustomTerms: (terms: string[]) => void;
  onAddDoc: (payload: { title: string; category: string; content: string }) => Promise<void>;
  onUploadFile: (payload: { filename: string; mimeType: string; base64Data: string }) => Promise<void>;
  onDeleteDoc: (id: string) => Promise<void>;
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const commaIdx = result.indexOf(',');
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.onerror = () => reject(new Error('Could not read the selected file.'));
    reader.readAsDataURL(file);
  });
}

export function PrivacyRagView({
  docs,
  customTerms,
  onChangeCustomTerms,
  onAddDoc,
  onUploadFile,
  onDeleteDoc,
}: PrivacyRagViewProps) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [showNoteForm, setShowNoteForm] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingNote, setSavingNote] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [termInput, setTermInput] = useState(customTerms.join(', '));

  const searchMatches = useMemo(() => {
    if (!searchQuery.trim()) return [];
    return rankLocalRagDocuments(searchQuery, docs, 4).hits.filter((h) => h.cosineSimilarity > 0.01);
  }, [searchQuery, docs]);

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;

    setErrorMessage(null);
    setStatusMessage(null);
    setUploading(true);

    try {
      for (const file of files) {
        if (file.size > 10 * 1024 * 1024) {
          throw new Error(`"${file.name}" is larger than 10 MB. Please upload a smaller file.`);
        }
        const base64Data = await readFileAsBase64(file);
        await onUploadFile({
          filename: file.name,
          mimeType: file.type || 'application/octet-stream',
          base64Data,
        });
      }
      setStatusMessage(
        files.length === 1
          ? `Added "${files[0].name}" to your private library.`
          : `Added ${files.length} files to your private library.`,
      );
    } catch (err) {
      setErrorMessage((err as Error).message);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleAddNoteSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) {
      setErrorMessage('Please enter both a title and text.');
      return;
    }
    setErrorMessage(null);
    setStatusMessage(null);
    setSavingNote(true);
    try {
      await onAddDoc({
        title: title.trim(),
        category: 'Note',
        content: content.trim(),
      });
      setTitle('');
      setContent('');
      setShowNoteForm(false);
      setStatusMessage('Note saved to your private library.');
    } catch (err) {
      setErrorMessage((err as Error).message);
    } finally {
      setSavingNote(false);
    }
  };

  const handleCustomTermsChange = (raw: string) => {
    setTermInput(raw);
    onChangeCustomTerms(
      raw
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length >= 2),
    );
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-20 md:pt-8 pb-16 space-y-8">
      <div className="border-b border-white/[0.08] pb-5">
        <h1 className="font-display text-2xl font-semibold text-white tracking-tight">
          Upload Docs
        </h1>
        <p className="mt-1 text-sm text-[#9893A8]">
          Upload PDFs, Word docs, PowerPoint slides, Excel sheets, images, or text notes so you can ask questions about them privately.
        </p>
      </div>

      {/* Primary File Upload Card */}
      <div className="rounded-2xl bg-[#110E19] border border-dashed border-violet-500/40 p-6 sm:p-8 text-center space-y-4">
        <div className="mx-auto h-11 w-11 rounded-xl bg-violet-600/15 border border-violet-500/30 flex items-center justify-center">
          <Upload className="h-5 w-5 text-violet-300" aria-hidden="true" />
        </div>

        <div className="space-y-1">
          <h2 className="font-display text-base font-semibold text-white">
            Choose files from your device
          </h2>
          <p className="text-xs text-[#9893A8]">
            Supports PDF, Word (.doc, .docx), PowerPoint (.ppt, .pptx), Excel (.xls, .xlsx, .csv), Images (.png, .jpg, .webp), and Text (.txt, .md)
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-3 pt-1">
          <label className="inline-flex items-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 px-5 py-2.5 text-xs font-medium text-white cursor-pointer transition-colors">
            <Upload className="h-4 w-4" aria-hidden="true" />
            <span>{uploading ? 'Uploading & Reading...' : 'Upload Files'}</span>
            <input
              type="file"
              multiple
              disabled={uploading}
              accept=".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.csv,.png,.jpg,.jpeg,.webp,.gif,.txt,.md,.json"
              onChange={handleFileUpload}
              className="sr-only"
            />
          </label>

          <button
            type="button"
            onClick={() => setShowNoteForm((v) => !v)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] px-4 py-2.5 text-xs font-medium text-[#CFCADF] transition-colors"
          >
            <Plus className="h-3.5 w-3.5 text-violet-400" aria-hidden="true" />
            <span>Paste Text Note</span>
            {showNoteForm ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        </div>

        {statusMessage && (
          <p className="text-xs text-emerald-300 pt-1" role="status">
            {statusMessage}
          </p>
        )}
        {errorMessage && (
          <p className="text-xs text-rose-400 pt-1" role="alert">
            {errorMessage}
          </p>
        )}
      </div>

      {/* Collapsible Manual Note Entry */}
      {showNoteForm && (
        <form
          onSubmit={handleAddNoteSubmit}
          className="rounded-xl bg-[#110E19] border border-white/[0.08] p-5 space-y-4"
        >
          <h2 className="font-display text-sm font-semibold text-white">Add Text Note</h2>
          <div>
            <label htmlFor="rag-doc-title" className="block text-xs text-[#A49FB6] mb-1">
              Title
            </label>
            <input
              id="rag-doc-title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Lecture Notes"
              className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
            />
          </div>
          <div>
            <label htmlFor="rag-doc-content" className="block text-xs text-[#A49FB6] mb-1">
              Content
            </label>
            <textarea
              id="rag-doc-content"
              rows={4}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Paste or type your notes..."
              className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
            />
          </div>
          <button
            type="submit"
            disabled={savingNote}
            className="rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-4 py-2 text-xs font-medium text-white transition-colors"
          >
            {savingNote ? 'Saving...' : 'Save Note'}
          </button>
        </form>
      )}

      {/* Quick Search Inside Uploaded Docs + Custom Hidden Words */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl bg-[#110E19] border border-white/[0.08] p-5 space-y-3">
          <h2 className="font-display text-sm font-semibold text-white">
            Find Inside Your Docs
          </h2>
          <div className="relative">
            <label htmlFor="rag-probe-input" className="sr-only">
              Search uploaded files
            </label>
            <Search className="h-3.5 w-3.5 text-[#8B85A0] absolute left-3 top-1/2 -translate-y-1/2" aria-hidden="true" />
            <input
              id="rag-probe-input"
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search keywords across your files..."
              className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
            />
          </div>

          {searchQuery.trim() && (
            <div className="space-y-2 pt-1">
              {searchMatches.length === 0 ? (
                <p className="text-xs text-[#8B85A0]">No matching documents found.</p>
              ) : (
                searchMatches.map((hit) => {
                  const pct = Math.min(99, Math.max(12, Math.round(hit.cosineSimilarity * 100)));
                  return (
                    <div
                      key={hit.docId}
                      className="rounded-lg bg-[#0B0912] border border-white/[0.06] p-3 flex items-center justify-between gap-3 text-xs"
                    >
                      <span className="font-medium text-white truncate">{hit.title}</span>
                      <span className="text-violet-300 shrink-0 tabular-nums">{pct}% match</span>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>

        <div className="rounded-xl bg-[#110E19] border border-white/[0.08] p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Lock className="h-4 w-4 text-violet-400" aria-hidden="true" />
            <h2 className="font-display text-sm font-semibold text-white">
              Words to Always Hide
            </h2>
          </div>
          <p className="text-xs text-[#9893A8]">
            Emails, phone numbers, and IDs are hidden automatically. Add any extra names or words to keep private:
          </p>
          <input
            id="custom-redact-input"
            type="text"
            value={termInput}
            onChange={(e) => handleCustomTermsChange(e.target.value)}
            placeholder="e.g. AcmeCorp, Dr. Patel, ProjectX"
            className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white placeholder-[#68627C] focus:outline-none focus:border-violet-500"
          />
        </div>
      </div>

      {/* Stored Documents */}
      <div className="rounded-xl bg-[#110E19] border border-white/[0.08] overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
          <h2 className="font-display text-sm font-semibold text-white">
            Your Uploaded Docs
          </h2>
          <span className="text-xs text-[#8B85A0] tabular-nums">{docs.length} files</span>
        </div>

        {docs.length === 0 ? (
          <div className="p-6 text-center text-sm text-[#8B85A0]">
            No documents uploaded yet.
          </div>
        ) : (
          <ul className="divide-y divide-white/[0.06]">
            {docs.map((d) => (
              <li key={d.id} className="flex items-center justify-between gap-4 px-5 py-3.5">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-xs font-medium text-white">
                    <FileText className="h-3.5 w-3.5 text-violet-400 shrink-0" aria-hidden="true" />
                    <span className="truncate">{d.title}</span>
                    <span className="text-[#78728F] font-normal shrink-0">
                      · {d.category} · <span className="tabular-nums">{d.tokenCount} words</span>
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onDeleteDoc(d.id)}
                  aria-label={`Delete '${d.title}'`}
                  className="p-1.5 rounded-lg text-[#8B85A0] hover:text-rose-400 transition-colors shrink-0"
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

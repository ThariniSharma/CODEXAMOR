import 'dotenv/config';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import {
  addFlNode,
  deleteRagDocument,
  deleteThreadById,
  getDb,
  getFlState,
  insertMessage,
  insertRagDocument,
  listRagDocuments,
  listThreads,
  saveFlRound,
  upsertThread,
} from './src/server/db.ts';
import { parseUploadedDocument } from './src/lib/docParser.ts';
import { runProductEvaluation } from './src/lib/evaluator.ts';
import { executeFederatedRound } from './src/lib/federated.ts';
import { fetchOpenWebResults, toSourceItems } from './src/lib/webSearch.ts';
import type { ChatMessage, FLNode, FocusMode, PrivacyAudit, RagSearchHit, SourceItem } from './src/lib/types.ts';

const PORT = Number(process.env.PORT) || 3000;

function getAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'web-source';
  }
}

function isRateLimitError(err: unknown): boolean {
  const raw = err instanceof Error ? err.message : String(err ?? '');
  return /429|RESOURCE_EXHAUSTED|quota|rate-?limit/i.test(raw);
}

function buildSystemInstruction(focusMode: FocusMode): string {
  const shared =
    'You are Codexamor, a helpful private research assistant. Answer clearly in clean Markdown. Place inline citations like [1], [2] right after claims backed by sources. Keep any masked placeholders such as [EMAIL_1] or [PHONE_1] exactly as written. End with "### Follow-up Questions" and 3 short bullet questions.';

  if (focusMode === 'research') {
    return `${shared} Provide thorough Deep Research: compare approaches, summarize evidence clearly, and finish with practical takeaways.`;
  }
  return `${shared} Combine retrieved web sources and any uploaded user document snippets into a clear, direct answer.`;
}

function splitAnswerAndFollowUps(rawText: string): { answer: string; followUps: string[] } {
  const marker = /###\s*Follow-?up Questions\s*\n([\s\S]*)$/i;
  const match = rawText.match(marker);
  if (!match) {
    return { answer: rawText.trim(), followUps: [] };
  }
  const answer = rawText.slice(0, match.index).trim();
  const followUps = match[1]
    .split('\n')
    .map((line) => line.replace(/^[-*\d.)\s]+/, '').trim())
    .filter((line) => line.length > 4)
    .slice(0, 3);
  return { answer, followUps };
}

function buildOfflineDocsResponse(query: string, hits: RagSearchHit[]): { answer: string; followUps: string[] } {
  if (!hits.length || hits[0].cosineSimilarity <= 0.01) {
    return {
      answer:
        `No matching sections were found in your uploaded documents for **"${query}"**.\n\n` +
        'Go to **Upload Docs** in the sidebar to add PDFs, Word docs, slides, spreadsheets, images, or notes.',
      followUps: [
        'Summarize my uploaded documents',
        'Explain how my personal details stay private',
      ],
    };
  }

  const sections = hits.map((hit, idx) => {
    const pct = Math.min(99, Math.max(10, Math.round(hit.cosineSimilarity * 100)));
    return `### ${hit.title} [${idx + 1}] (${pct}% match)\n${hit.excerpt}`;
  });

  return {
    answer: `Found **${hits.length}** matching document(s) in your private library:\n\n${sections.join('\n\n')}`,
    followUps: hits.slice(0, 2).map((h) => `Tell me more about ${h.title}`),
  };
}

async function runWebAndAiSearch(params: {
  effectiveQuery: string;
  focusMode: FocusMode;
  relevantRag: RagSearchHit[];
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  sources: SourceItem[];
}): Promise<{ answerText: string; followUps: string[] }> {
  const { effectiveQuery, focusMode, relevantRag, history, sources } = params;
  const ai = getAiClient();

  const ragBlock =
    relevantRag.length > 0
      ? `\n\nUploaded Document Context:\n${relevantRag
          .map((h, i) => `[${i + 1}] ${h.title}: ${h.excerpt}`)
          .join('\n')}`
      : '';

  const historyBlock =
    history.length > 0
      ? `Previous conversation (private details masked):\n${history
          .slice(-4)
          .map((m) => `${m.role.toUpperCase()}: ${m.content.slice(0, 400)}`)
          .join('\n\n')}\n\n`
      : '';

  // 1. Primary path: Gemini + Google Search Grounding
  if (ai) {
    try {
      const thinkingLevel = focusMode === 'research' ? ThinkingLevel.HIGH : ThinkingLevel.LOW;
      const aiResponse = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `${historyBlock}Current Question: ${effectiveQuery}${ragBlock}`,
        config: {
          systemInstruction: buildSystemInstruction(focusMode),
          thinkingConfig: { thinkingLevel },
          tools: [{ googleSearch: {} }],
        },
      });

      const chunks = aiResponse.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [];
      const seenUrls = new Set<string>();
      for (const chunk of chunks) {
        const web = (chunk as { web?: { uri?: string; title?: string } }).web;
        if (web?.uri && !seenUrls.has(web.uri)) {
          seenUrls.add(web.uri);
          sources.push({
            id: sources.length + 1,
            title: web.title || extractDomain(web.uri),
            url: web.uri,
            domain: extractDomain(web.uri),
            snippet: '',
            origin: 'web',
          });
        }
      }

      const parsed = splitAnswerAndFollowUps(aiResponse.text ?? '');
      return {
        answerText: parsed.answer || 'No response content generated.',
        followUps: parsed.followUps,
      };
    } catch (err) {
      if (!isRateLimitError(err)) {
        throw err;
      }
      // Google Search Grounding hit rate limit: continue to live open-web retrieval + lighter model
    }
  }

  // 2. Resilient fallback: Fetch real live web snippets (Wikipedia, DuckDuckGo, arXiv)
  const webSnippets = await fetchOpenWebResults(effectiveQuery, focusMode === 'research');
  const addedWebSources = toSourceItems(webSnippets, sources.length + 1);
  sources.push(...addedWebSources);

  const webContext =
    addedWebSources.length > 0
      ? `\n\nLive Web Sources:\n${addedWebSources
          .map((s) => `[${s.id}] ${s.title} (${s.domain}): ${s.snippet}`)
          .join('\n')}`
      : '';

  if (ai) {
    for (const model of ['gemini-3.1-flash-lite', 'gemini-4-hydrogen']) {
      try {
        const aiResponse = await ai.models.generateContent({
          model,
          contents: `${historyBlock}Current Question: ${effectiveQuery}${ragBlock}${webContext}`,
          config: {
            systemInstruction: buildSystemInstruction(focusMode),
          },
        });
        const parsed = splitAnswerAndFollowUps(aiResponse.text ?? '');
        return {
          answerText: parsed.answer || 'No response content generated.',
          followUps: parsed.followUps,
        };
      } catch (err) {
        if (!isRateLimitError(err)) {
          throw err;
        }
      }
    }
  }

  // 3. If all Gemini API models are currently rate-limited, still present real retrieved web & doc sources transparently
  if (addedWebSources.length > 0 || relevantRag.length > 0) {
    const notice =
      '> **Note:** Live AI summary generation reached the upstream API rate limit, so retrieved web and document matches are shown directly below.\n\n';
    const webMd = addedWebSources
      .map((s) => `### ${s.title} [${s.id}]\n${s.snippet}\n\n[Open source](${s.url})`)
      .join('\n\n');
    const docMd =
      relevantRag.length > 0
        ? `\n\n---\n\n### From Your Uploaded Docs\n${relevantRag
            .map((h, i) => `- **${h.title} [${i + 1}]**: ${h.excerpt}`)
            .join('\n')}`
        : '';

    return {
      answerText: `${notice}${webMd}${docMd}`.trim(),
      followUps: [`Search my uploaded docs for ${effectiveQuery}`],
    };
  }

  throw new Error('Web search reached its temporary upstream API limit. Please try again shortly or switch to My Docs.');
}

async function startServer(): Promise<void> {
  await getDb();

  const app = express();
  app.use(express.json({ limit: '15mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, service: 'codexamor-api' });
  });

  app.get('/api/evaluate', (_req, res) => {
    try {
      res.json(runProductEvaluation());
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.get('/api/threads', async (_req, res) => {
    try {
      const threads = await listThreads();
      res.json({ threads });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.delete('/api/threads/:id', async (req, res) => {
    try {
      await deleteThreadById(req.params.id);
      res.json({ ok: true });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post('/api/search', async (req, res) => {
    try {
      const {
        threadId: rawThreadId,
        query,
        sanitizedQuery,
        focusMode = 'all',
        localRagHits = [],
        privacyAudit,
        history = [],
      } = req.body as {
        threadId?: string;
        query: string;
        sanitizedQuery?: string;
        focusMode?: FocusMode;
        localRagHits?: RagSearchHit[];
        privacyAudit?: PrivacyAudit;
        history?: Array<{ role: 'user' | 'assistant'; content: string }>;
      };

      const effectiveQuery = (sanitizedQuery ?? query ?? '').trim();
      if (!effectiveQuery) {
        res.status(400).json({ error: 'Please enter a question to search.' });
        return;
      }

      const sources: SourceItem[] = [];
      const relevantRag = (localRagHits ?? []).filter((h) => h.cosineSimilarity > 0.02);
      relevantRag.forEach((hit) => {
        sources.push({
          id: sources.length + 1,
          title: hit.title,
          url: `#doc-${hit.docId}`,
          domain: `My Docs · ${hit.category}`,
          snippet: hit.excerpt,
          origin: 'local_rag',
          similarity: hit.cosineSimilarity,
        });
      });

      let answerText = '';
      let followUps: string[] = [];

      if (focusMode === 'rag') {
        const offline = buildOfflineDocsResponse(effectiveQuery, relevantRag);
        answerText = offline.answer;
        followUps = offline.followUps;
      } else {
        const result = await runWebAndAiSearch({
          effectiveQuery,
          focusMode,
          relevantRag,
          history,
          sources,
        });
        answerText = result.answerText;
        followUps = result.followUps;
      }

      const threadId = rawThreadId || crypto.randomUUID();
      const now = new Date().toISOString();
      const threadTitle = (query || effectiveQuery).slice(0, 68);
      await upsertThread({ id: threadId, title: threadTitle, focusMode });

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        threadId,
        role: 'user',
        content: query,
        focusMode,
        sources: [],
        followUps: [],
        privacyAudit,
        createdAt: now,
      };
      await insertMessage(userMsg);

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        threadId,
        role: 'assistant',
        content: answerText,
        focusMode,
        sources,
        followUps,
        createdAt: new Date(Date.now() + 5).toISOString(),
      };
      await insertMessage(assistantMsg);

      res.json({
        threadId,
        userMessage: userMsg,
        assistantMessage: assistantMsg,
      });
    } catch (err) {
      const status = isRateLimitError(err) ? 429 : 500;
      const cleanMessage = isRateLimitError(err)
        ? 'Web search is temporarily busy due to API limits. Please try again shortly or switch to My Docs.'
        : (err instanceof Error ? err.message : 'Search request failed.');
      res.status(status).json({ error: cleanMessage });
    }
  });

  app.get('/api/fl/state', async (_req, res) => {
    try {
      const state = await getFlState();
      res.json(state);
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post('/api/fl/nodes', async (req, res) => {
    try {
      const { name, organization, domain, architecture = 'ANN_MLP', sampleCount = 500 } = req.body;
      if (!name || !String(name).trim() || !organization || !String(organization).trim()) {
        res.status(400).json({ error: 'Name and organization are required.' });
        return;
      }

      const newNode: FLNode = {
        id: `node-${crypto.randomUUID().slice(0, 8)}`,
        name: String(name).trim(),
        organization: String(organization).trim(),
        domain: String(domain || 'General Knowledge').trim(),
        architecture: architecture === 'SMALL_LSTM_GRU' ? 'SMALL_LSTM_GRU' : 'ANN_MLP',
        sampleCount: Math.max(50, Number(sampleCount) || 500),
        epsilon: 0,
        delta: 1e-5,
        localAccuracy: 0.74,
        localLoss: 0.64,
        status: 'active',
        joinedAt: new Date().toISOString(),
      };
      await addFlNode(newNode);
      const state = await getFlState();
      res.status(201).json({ node: newNode, ...state });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post('/api/fl/rounds', async (req, res) => {
    try {
      const {
        simulatedSwarmSize = 100,
        architecture = 'SMALL_LSTM_GRU',
        aggregationStrategy = 'FedAvg',
        epsilonPerRound = 1.2,
        clipNorm = 1.0,
      } = req.body;

      const { nodes, rounds } = await getFlState();
      const { round, updatedNodes } = executeFederatedRound({
        nodes,
        previousRound: rounds[rounds.length - 1],
        simulatedSwarmSize: Number(simulatedSwarmSize),
        architecture,
        aggregationStrategy,
        epsilonPerRound: Number(epsilonPerRound),
        clipNorm: Number(clipNorm),
      });

      await saveFlRound(round, updatedNodes);
      const nextState = await getFlState();
      res.status(201).json({ round, ...nextState });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.get('/api/rag/docs', async (_req, res) => {
    try {
      const docs = await listRagDocuments();
      res.json({ docs });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post('/api/rag/docs', async (req, res) => {
    try {
      const { title, category = 'Note', content } = req.body;
      if (!title || !String(title).trim() || !content || !String(content).trim()) {
        res.status(400).json({ error: 'Document title and text are required.' });
        return;
      }
      const cleanContent = String(content).trim();
      const doc = {
        id: `doc-${crypto.randomUUID().slice(0, 8)}`,
        title: String(title).trim(),
        category: String(category).trim(),
        content: cleanContent,
        tokenCount: cleanContent.split(/\s+/).filter(Boolean).length,
        createdAt: new Date().toISOString(),
      };
      await insertRagDocument(doc);
      const docs = await listRagDocuments();
      res.status(201).json({ doc, docs });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  app.post('/api/rag/upload', async (req, res) => {
    try {
      const { filename, mimeType, base64Data, title, notes } = req.body as {
        filename?: string;
        mimeType?: string;
        base64Data?: string;
        title?: string;
        notes?: string;
      };
      if (!filename || !base64Data) {
        res.status(400).json({ error: 'Please select a valid file to upload.' });
        return;
      }
      const parsed = await parseUploadedDocument({
        filename,
        mimeType,
        base64Data,
        title,
        notes,
      });
      const doc = {
        id: `doc-${crypto.randomUUID().slice(0, 8)}`,
        title: parsed.title,
        category: parsed.fileType,
        content: parsed.content,
        tokenCount: parsed.content.split(/\s+/).filter(Boolean).length,
        createdAt: new Date().toISOString(),
      };
      await insertRagDocument(doc);
      const docs = await listRagDocuments();
      res.status(201).json({ doc, docs });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message || 'Could not process the uploaded file.' });
    }
  });

  app.delete('/api/rag/docs/:id', async (req, res) => {
    try {
      await deleteRagDocument(req.params.id);
      const docs = await listRagDocuments();
      res.json({ ok: true, docs });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  const distDir = path.resolve(process.cwd(), 'dist');
  const isProd =
    fs.existsSync(distDir) &&
    (process.env.NODE_ENV === 'production' ||
      process.env.K_SERVICE !== undefined ||
      process.env.npm_lifecycle_event === 'start');

  if (isProd) {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Codexamor server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start Codexamor server:', err);
  process.exit(1);
});

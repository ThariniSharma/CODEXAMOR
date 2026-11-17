import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { AboutProjectView } from './components/AboutProjectView';
import { FederatedHubView } from './components/FederatedHubView';
import { PrivacyRagView } from './components/PrivacyRagView';
import { SearchComposer } from './components/SearchComposer';
import { Sidebar, type ActiveTab } from './components/Sidebar';
import { ThreadView } from './components/ThreadView';
import { redactPii } from './lib/privacy';
import { rankLocalRagDocuments } from './lib/tfidfRag';
import type {
  FLNode,
  FLRound,
  FocusMode,
  LocalRagDocument,
  ModelArchitecture,
  Thread,
} from './lib/types';

const STARTER_PROMPTS: Array<{ title: string; mode: FocusMode }> = [
  {
    title: 'Summarize my uploaded documents',
    mode: 'rag',
  },
  {
    title: 'How does federated learning improve AI without sharing personal data?',
    mode: 'research',
  },
  {
    title: 'Latest privacy-preserving AI breakthroughs',
    mode: 'all',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('search');
  const [mobileOpen, setMobileOpen] = useState(false);

  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);

  const [query, setQuery] = useState('');
  const [followUpQuery, setFollowUpQuery] = useState('');
  const [focusMode, setFocusMode] = useState<FocusMode>('all');
  const [shieldEnabled, setShieldEnabled] = useState(true);
  const [customTerms, setCustomTerms] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [flNodes, setFlNodes] = useState<FLNode[]>([]);
  const [flRounds, setFlRounds] = useState<FLRound[]>([]);
  const [ragDocs, setRagDocs] = useState<LocalRagDocument[]>([]);

  const loadInitialState = useCallback(async () => {
    try {
      const [threadsRes, flRes, ragRes] = await Promise.all([
        fetch('/api/threads'),
        fetch('/api/fl/state'),
        fetch('/api/rag/docs'),
      ]);
      if (threadsRes.ok) {
        const data = await threadsRes.json();
        setThreads(data.threads ?? []);
      }
      if (flRes.ok) {
        const data = await flRes.json();
        setFlNodes(data.nodes ?? []);
        setFlRounds(data.rounds ?? []);
      }
      if (ragRes.ok) {
        const data = await ragRes.json();
        setRagDocs(data.docs ?? []);
      }
    } catch {
      // Initial load failure will surface if user interacts
    }
  }, []);

  useEffect(() => {
    void loadInitialState();
  }, [loadInitialState]);

  const activeThread = useMemo(
    () => threads.find((t) => t.id === activeThreadId) ?? null,
    [threads, activeThreadId],
  );

  const executeSearch = async (rawText: string, modeOverride?: FocusMode) => {
    const trimmed = rawText.trim();
    if (!trimmed || isSearching) return;

    const effectiveMode = modeOverride ?? focusMode;
    setSearchError(null);
    setIsSearching(true);

    const audit = redactPii(trimmed, customTerms, shieldEnabled, 0.25);
    const { hits: localRagHits } = rankLocalRagDocuments(audit.sanitizedQuery, ragDocs, 3);

    const history = (activeThread?.messages ?? []).map((m) => ({
      role: m.role,
      content: m.privacyAudit?.sanitizedQuery ?? m.content,
    }));

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threadId: activeThreadId ?? undefined,
          query: trimmed,
          sanitizedQuery: audit.sanitizedQuery,
          focusMode: effectiveMode,
          localRagHits,
          privacyAudit: audit,
          history,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Search failed.');
      }

      const refreshed = await fetch('/api/threads');
      if (refreshed.ok) {
        const tData = await refreshed.json();
        setThreads(tData.threads ?? []);
      }

      setActiveThreadId(data.threadId);
      setActiveTab('search');
      setQuery('');
      setFollowUpQuery('');
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Unable to reach search service.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleDeleteThread = async (id: string) => {
    await fetch(`/api/threads/${encodeURIComponent(id)}`, { method: 'DELETE' });
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (activeThreadId === id) {
      setActiveThreadId(null);
    }
  };

  const handleRegisterNode = async (payload: {
    name: string;
    organization: string;
    domain: string;
    architecture: ModelArchitecture;
    sampleCount: number;
  }) => {
    const res = await fetch('/api/fl/nodes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to join network');
    setFlNodes(data.nodes ?? []);
    setFlRounds(data.rounds ?? []);
  };

  const handleRunRound = async (payload: {
    simulatedSwarmSize: number;
    architecture: ModelArchitecture;
    aggregationStrategy: 'FedAvg' | 'FedProx';
    epsilonPerRound: number;
    clipNorm: number;
  }) => {
    const res = await fetch('/api/fl/rounds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to run round');
    setFlNodes(data.nodes ?? []);
    setFlRounds(data.rounds ?? []);
  };

  const handleAddRagDoc = async (payload: { title: string; category: string; content: string }) => {
    const res = await fetch('/api/rag/docs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to save document');
    setRagDocs(data.docs ?? []);
  };

  const handleUploadFile = async (payload: { filename: string; mimeType: string; base64Data: string }) => {
    const res = await fetch('/api/rag/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to upload file');
    setRagDocs(data.docs ?? []);
  };

  const handleDeleteRagDoc = async (id: string) => {
    const res = await fetch(`/api/rag/docs/${encodeURIComponent(id)}`, { method: 'DELETE' });
    const data = await res.json();
    if (res.ok) {
      setRagDocs(data.docs ?? []);
    }
  };

  return (
    <div className="min-h-screen flex bg-[#08070B] text-[#E6E3F0]">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={(id) => {
          setActiveThreadId(id || null);
          setSearchError(null);
        }}
        onNewThread={() => {
          setActiveTab('search');
          setActiveThreadId(null);
          setSearchError(null);
        }}
        onDeleteThread={handleDeleteThread}
        shieldEnabled={shieldEnabled}
        mobileOpen={mobileOpen}
        onToggleMobile={setMobileOpen}
      />

      <main className="flex-1 min-w-0">
        {activeTab === 'search' && activeThread ? (
          <ThreadView
            thread={activeThread}
            followUpQuery={followUpQuery}
            onChangeFollowUpQuery={setFollowUpQuery}
            focusMode={focusMode}
            onChangeFocusMode={setFocusMode}
            shieldEnabled={shieldEnabled}
            onToggleShield={setShieldEnabled}
            customTerms={customTerms}
            onSubmitQuery={(override) => executeSearch(override ?? followUpQuery)}
            isLoading={isSearching}
            errorMessage={searchError}
          />
        ) : activeTab === 'search' ? (
          <div className="min-h-screen flex flex-col justify-center max-w-2xl mx-auto px-4 sm:px-6 pt-20 md:pt-8 pb-12">
            <div className="space-y-6">
              <h1 className="font-display text-3xl sm:text-4xl font-semibold text-white tracking-tight text-center text-balance">
                Ask anything, privately
              </h1>

              <SearchComposer
                query={query}
                onChangeQuery={setQuery}
                focusMode={focusMode}
                onChangeFocusMode={setFocusMode}
                shieldEnabled={shieldEnabled}
                onToggleShield={setShieldEnabled}
                customTerms={customTerms}
                onSubmit={() => executeSearch(query)}
                isLoading={isSearching}
              />

              {isSearching && (
                <div className="text-center text-xs text-violet-300 animate-pulse" aria-live="polite">
                  Searching...
                </div>
              )}

              {searchError && (
                <div
                  role="alert"
                  className="rounded-xl bg-rose-950/30 border border-rose-500/30 px-4 py-3 text-xs text-rose-200"
                >
                  {searchError}
                </div>
              )}

              <div className="divide-y divide-white/[0.06] border-t border-b border-white/[0.06]">
                {STARTER_PROMPTS.map((item) => (
                  <button
                    key={item.title}
                    type="button"
                    disabled={isSearching}
                    onClick={() => {
                      setFocusMode(item.mode);
                      void executeSearch(item.title, item.mode);
                    }}
                    className="w-full flex items-center justify-between gap-4 py-3 text-left text-xs sm:text-sm text-[#B8B2CC] hover:text-white transition-colors"
                  >
                    <span className="truncate">{item.title}</span>
                    <ArrowRight className="h-3.5 w-3.5 shrink-0 text-violet-400" aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : activeTab === 'federated' ? (
          <FederatedHubView
            nodes={flNodes}
            rounds={flRounds}
            onRegisterNode={handleRegisterNode}
            onRunRound={handleRunRound}
          />
        ) : activeTab === 'privacy_rag' ? (
          <PrivacyRagView
            docs={ragDocs}
            customTerms={customTerms}
            onChangeCustomTerms={setCustomTerms}
            onAddDoc={handleAddRagDoc}
            onUploadFile={handleUploadFile}
            onDeleteDoc={handleDeleteRagDoc}
          />
        ) : (
          <AboutProjectView />
        )}
      </main>
    </div>
  );
}

import { useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowRight,
  Check,
  Copy,
  ExternalLink,
  Eye,
  EyeOff,
  FileText,
  Globe,
  Layers,
  Lock,
} from 'lucide-react';
import { restoreLocalPii } from '../lib/privacy';
import type { ChatMessage, FocusMode, Thread } from '../lib/types';
import { SearchComposer } from './SearchComposer';

interface ThreadViewProps {
  thread: Thread;
  followUpQuery: string;
  onChangeFollowUpQuery: (val: string) => void;
  focusMode: FocusMode;
  onChangeFocusMode: (mode: FocusMode) => void;
  shieldEnabled: boolean;
  onToggleShield: (enabled: boolean) => void;
  customTerms: string[];
  onSubmitQuery: (overrideQuery?: string) => void;
  isLoading: boolean;
  errorMessage: string | null;
}

function TurnPair({
  userMsg,
  assistantMsg,
  onSelectFollowUp,
}: {
  userMsg: ChatMessage;
  assistantMsg?: ChatMessage;
  onSelectFollowUp: (q: string) => void;
}) {
  const [restorePii, setRestorePii] = useState(false);
  const [copied, setCopied] = useState(false);

  const redactions = userMsg.privacyAudit?.redactions ?? [];
  const rawAssistantText = assistantMsg?.content ?? '';
  const displayedAssistantText =
    restorePii && redactions.length > 0 ? restoreLocalPii(rawAssistantText, redactions) : rawAssistantText;

  const handleCopy = async () => {
    if (!displayedAssistantText) return;
    await navigator.clipboard.writeText(displayedAssistantText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <article className="pb-10 border-b border-white/[0.06] last:border-b-0">
      {/* User Query Heading */}
      <h2 className="font-display text-xl sm:text-2xl font-semibold text-white tracking-tight leading-snug mb-2.5 text-balance">
        {userMsg.content}
      </h2>

      {/* Friendly Privacy Confirmation */}
      {userMsg.privacyAudit?.shieldEnabled && (
        <div className="mb-5 flex items-center gap-2 text-xs text-violet-300">
          <Lock className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>
            {redactions.length > 0
              ? 'Personal details were kept private during search'
              : 'Protected private search'}
          </span>
        </div>
      )}

      {/* Assistant Sources + Answer */}
      {assistantMsg && (
        <div className="space-y-6">
          {/* Sources Strip */}
          {assistantMsg.sources.length > 0 && (
            <section aria-label="Sources">
              <div className="flex items-center gap-2 text-xs font-medium text-[#A49FB6] mb-2.5">
                <Layers className="h-3.5 w-3.5 text-violet-400" aria-hidden="true" />
                <span>Sources</span>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">{assistantMsg.sources.length}</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                {assistantMsg.sources.map((src) => (
                  <a
                    key={src.id}
                    href={src.origin === 'web' ? src.url : undefined}
                    target={src.origin === 'web' ? '_blank' : undefined}
                    rel={src.origin === 'web' ? 'noopener noreferrer' : undefined}
                    className="group rounded-lg bg-[#120F1B] hover:bg-[#171324] border border-white/[0.08] hover:border-violet-500/40 p-3 transition-colors flex flex-col justify-between gap-2"
                  >
                    <div className="text-xs font-medium text-[#E6E3F0] line-clamp-2 leading-snug group-hover:text-white">
                      {src.title}
                    </div>
                    <div className="flex items-center justify-between gap-2 text-[11px] text-[#8B85A0]">
                      <span className="flex items-center gap-1.5 truncate">
                        {src.origin === 'local_rag' ? (
                          <FileText className="h-3 w-3 text-violet-400 shrink-0" aria-hidden="true" />
                        ) : (
                          <Globe className="h-3 w-3 text-violet-400 shrink-0" aria-hidden="true" />
                        )}
                        <span className="truncate">{src.domain}</span>
                      </span>
                      <span className="font-mono tabular-nums text-violet-300 shrink-0 flex items-center gap-1">
                        [{src.id}]
                        {src.origin === 'web' && <ExternalLink className="h-2.5 w-2.5" aria-hidden="true" />}
                      </span>
                    </div>
                  </a>
                ))}
              </div>
            </section>
          )}

          {/* Answer Prose */}
          <section aria-label="Answer">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-xs font-medium text-violet-300">Answer</span>
              <div className="flex items-center gap-2">
                {redactions.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setRestorePii((v) => !v)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-[#C4BFE0] hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-colors"
                  >
                    {restorePii ? (
                      <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
                    ) : (
                      <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                    <span>{restorePii ? 'Hide Details' : 'Show My Details'}</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleCopy}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-[#A49FB6] hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-colors"
                >
                  {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            </div>

            <div className="prose-answer">
              <Markdown remarkPlugins={[remarkGfm]}>{displayedAssistantText}</Markdown>
            </div>
          </section>

          {/* Follow-up Questions */}
          {assistantMsg.followUps.length > 0 && (
            <section aria-label="Related follow-up questions" className="pt-2">
              <div className="text-xs font-medium text-[#9893A8] mb-2">Related</div>
              <div className="divide-y divide-white/[0.06] border-t border-b border-white/[0.06]">
                {assistantMsg.followUps.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => onSelectFollowUp(q)}
                    className="w-full flex items-center justify-between gap-4 py-2.5 text-left text-sm text-[#CFCADF] hover:text-violet-300 transition-colors"
                  >
                    <span>{q}</span>
                    <ArrowRight className="h-4 w-4 shrink-0 text-violet-400" aria-hidden="true" />
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </article>
  );
}

export function ThreadView({
  thread,
  followUpQuery,
  onChangeFollowUpQuery,
  focusMode,
  onChangeFocusMode,
  shieldEnabled,
  onToggleShield,
  customTerms,
  onSubmitQuery,
  isLoading,
  errorMessage,
}: ThreadViewProps) {
  const pairs: Array<{ user: ChatMessage; assistant?: ChatMessage }> = [];
  for (let i = 0; i < thread.messages.length; i++) {
    const m = thread.messages[i];
    if (m.role === 'user') {
      const next = thread.messages[i + 1];
      pairs.push({
        user: m,
        assistant: next?.role === 'assistant' ? next : undefined,
      });
    }
  }

  return (
    <div className="flex flex-col min-h-full">
      <div className="flex-1 max-w-3xl w-full mx-auto px-4 sm:px-6 pt-20 md:pt-8 pb-36 space-y-10">
        {pairs.map((p) => (
          <TurnPair
            key={p.user.id}
            userMsg={p.user}
            assistantMsg={p.assistant}
            onSelectFollowUp={(q) => onSubmitQuery(q)}
          />
        ))}

        {isLoading && (
          <div className="space-y-3 py-4" aria-live="polite" aria-label="Searching and writing answer">
            <div className="h-4 w-44 rounded bg-violet-500/20 animate-pulse" />
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              <div className="h-16 rounded-lg bg-white/[0.04] animate-pulse" />
              <div className="h-16 rounded-lg bg-white/[0.04] animate-pulse" />
              <div className="h-16 rounded-lg bg-white/[0.04] animate-pulse" />
            </div>
            <div className="h-4 w-full rounded bg-white/[0.05] animate-pulse" />
            <div className="h-4 w-5/6 rounded bg-white/[0.05] animate-pulse" />
            <div className="h-4 w-2/3 rounded bg-white/[0.05] animate-pulse" />
          </div>
        )}

        {errorMessage && (
          <div
            role="alert"
            className="rounded-xl bg-rose-950/30 border border-rose-500/30 px-4 py-3 text-sm text-rose-200"
          >
            {errorMessage}
          </div>
        )}
      </div>

      {/* Sticky Bottom Follow-up Composer */}
      <div className="sticky bottom-0 z-20 bg-gradient-to-t from-[#08070B] via-[#08070B]/95 to-transparent pt-4 pb-4 px-4 sm:px-6">
        <div className="max-w-3xl mx-auto">
          <SearchComposer
            query={followUpQuery}
            onChangeQuery={onChangeFollowUpQuery}
            focusMode={focusMode}
            onChangeFocusMode={onChangeFocusMode}
            shieldEnabled={shieldEnabled}
            onToggleShield={onToggleShield}
            customTerms={customTerms}
            onSubmit={() => onSubmitQuery()}
            isLoading={isLoading}
            compact
            placeholder="Ask a follow-up..."
          />
        </div>
      </div>
    </div>
  );
}

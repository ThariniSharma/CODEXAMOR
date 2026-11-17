import { useMemo, type KeyboardEvent } from 'react';
import { ArrowUp, FileText, Globe, Lock, Shield, ShieldOff, Sparkles } from 'lucide-react';
import { redactPii } from '../lib/privacy';
import type { FocusMode } from '../lib/types';

interface SearchComposerProps {
  query: string;
  onChangeQuery: (value: string) => void;
  focusMode: FocusMode;
  onChangeFocusMode: (mode: FocusMode) => void;
  shieldEnabled: boolean;
  onToggleShield: (enabled: boolean) => void;
  customTerms: string[];
  onSubmit: () => void;
  isLoading: boolean;
  compact?: boolean;
  placeholder?: string;
}

const FOCUS_MODES: Array<{ id: FocusMode; label: string; icon: typeof Globe; hint: string }> = [
  { id: 'all', label: 'Web Search', icon: Globe, hint: 'Search the web and your uploaded files' },
  { id: 'research', label: 'Deep Research', icon: Sparkles, hint: 'Detailed research with broader citations' },
  { id: 'rag', label: 'My Docs', icon: FileText, hint: 'Search only inside your uploaded documents' },
];

export function SearchComposer({
  query,
  onChangeQuery,
  focusMode,
  onChangeFocusMode,
  shieldEnabled,
  onToggleShield,
  customTerms,
  onSubmit,
  isLoading,
  compact = false,
  placeholder = 'Ask anything privately...',
}: SearchComposerProps) {
  const liveAudit = useMemo(
    () => redactPii(query, customTerms, shieldEnabled),
    [query, customTerms, shieldEnabled],
  );

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className="w-full rounded-2xl bg-[#120F1B] border border-white/[0.1] focus-within:border-violet-500/55 transition-colors">
      <div className={compact ? 'px-4 pt-3 pb-2' : 'px-4 pt-4 pb-2.5'}>
        <label htmlFor={compact ? 'followup-query-input' : 'main-query-input'} className="sr-only">
          {placeholder}
        </label>
        <textarea
          id={compact ? 'followup-query-input' : 'main-query-input'}
          rows={compact ? 1 : 2}
          value={query}
          onChange={(e) => onChangeQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          className="w-full resize-none bg-transparent text-[15px] text-white placeholder-[#78728F] focus:outline-none leading-relaxed"
        />

        {shieldEnabled && liveAudit.redactions.length > 0 && (
          <div className="mt-2 inline-flex items-center gap-1.5 text-xs text-violet-300">
            <Lock className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>Personal details will stay private before searching</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-1.5">
        <div className="flex items-center gap-1" role="group" aria-label="Search mode">
          {FOCUS_MODES.map((m) => {
            const Icon = m.icon;
            const active = focusMode === m.id;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => onChangeFocusMode(m.id)}
                title={m.hint}
                aria-pressed={active}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 ${
                  active
                    ? 'bg-violet-600/20 text-violet-200 border border-violet-500/35'
                    : 'text-[#9893A8] hover:text-white hover:bg-white/[0.05] border border-transparent'
                }`}
              >
                <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>{m.label}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2 shrink-0 ml-auto">
          <button
            type="button"
            onClick={() => onToggleShield(!shieldEnabled)}
            title="Automatically hides emails, phone numbers, and IDs before sending a search"
            aria-pressed={shieldEnabled}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              shieldEnabled
                ? 'text-violet-300 hover:bg-violet-500/10'
                : 'text-[#7D7794] hover:text-white hover:bg-white/[0.05]'
            }`}
          >
            {shieldEnabled ? (
              <Shield className="h-3.5 w-3.5 text-violet-400" aria-hidden="true" />
            ) : (
              <ShieldOff className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            <span>{shieldEnabled ? 'Privacy On' : 'Privacy Off'}</span>
          </button>

          <button
            type="button"
            onClick={onSubmit}
            disabled={!query.trim() || isLoading}
            aria-label="Submit search query"
            className="flex items-center justify-center h-8 w-8 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:bg-violet-600/25 disabled:text-white/35 text-white transition-colors shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
          >
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}

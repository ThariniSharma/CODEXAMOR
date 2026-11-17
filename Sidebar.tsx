import {
  FileUp,
  Info,
  Lock,
  Menu,
  MessageSquare,
  Plus,
  Search,
  ShieldCheck,
  Trash2,
  Users,
  X,
} from 'lucide-react';
import type { Thread } from '../lib/types';

export type ActiveTab = 'search' | 'federated' | 'privacy_rag' | 'about';

interface SidebarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  threads: Thread[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onNewThread: () => void;
  onDeleteThread: (threadId: string) => void;
  shieldEnabled: boolean;
  mobileOpen: boolean;
  onToggleMobile: (open: boolean) => void;
}

export function Sidebar({
  activeTab,
  onSelectTab,
  threads,
  activeThreadId,
  onSelectThread,
  onNewThread,
  onDeleteThread,
  shieldEnabled,
  mobileOpen,
  onToggleMobile,
}: SidebarProps) {
  const navItems: Array<{ id: ActiveTab; label: string; icon: typeof Search }> = [
    { id: 'search', label: 'Search', icon: Search },
    { id: 'privacy_rag', label: 'Upload Docs', icon: FileUp },
    { id: 'federated', label: 'Network', icon: Users },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  const sidebarContent = (
    <div className="flex h-full flex-col bg-[#0B0911] border-r border-white/[0.07] text-[#E6E3F0]">
      {/* Brand Zone */}
      <div className="flex items-center justify-between px-4 py-4">
        <button
          type="button"
          onClick={onNewThread}
          className="flex items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 rounded-md"
        >
          <ShieldCheck className="h-5 w-5 text-violet-400 shrink-0" aria-hidden="true" />
          <span className="font-display text-lg font-semibold tracking-tight text-white whitespace-nowrap shrink-0">
            codexamor
          </span>
        </button>

        <button
          type="button"
          onClick={() => onToggleMobile(false)}
          aria-label="Close navigation menu"
          className="md:hidden p-2 rounded-lg text-[#9893A8] hover:text-white hover:bg-white/[0.06]"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Primary Action */}
      <div className="px-3 pb-3">
        <button
          type="button"
          onClick={() => {
            onNewThread();
            onToggleMobile(false);
          }}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white px-3.5 py-2 text-xs font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          <Plus className="h-4 w-4" aria-hidden="true" />
          <span>New Thread</span>
        </button>
      </div>

      {/* Navigation */}
      <nav aria-label="Primary sections" className="px-3 space-y-0.5">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isCurrent = activeTab === item.id && (item.id !== 'search' || activeThreadId === null);
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelectTab(item.id);
                if (item.id === 'search') {
                  onSelectThread('');
                }
                onToggleMobile(false);
              }}
              aria-current={isCurrent ? 'page' : undefined}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 ${
                activeTab === item.id
                  ? 'bg-violet-600/15 text-violet-200 font-medium'
                  : 'text-[#9E99B0] hover:text-white hover:bg-white/[0.04]'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Library Threads */}
      <div className="mt-5 px-3 flex-1 overflow-y-auto min-h-0">
        {threads.length > 0 && (
          <>
            <div className="px-3 pb-1.5 text-xs text-[#736D88]">History</div>
            <ul className="space-y-0.5">
              {threads.map((thread) => {
                const selected = activeTab === 'search' && activeThreadId === thread.id;
                return (
                  <li key={thread.id} className="group flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        onSelectThread(thread.id);
                        onSelectTab('search');
                        onToggleMobile(false);
                      }}
                      className={`flex-1 min-w-0 flex items-center gap-2 px-3 py-2 rounded-lg text-left text-xs transition-colors ${
                        selected
                          ? 'bg-white/[0.07] text-white font-medium'
                          : 'text-[#9E99B0] hover:text-white hover:bg-white/[0.04]'
                      }`}
                    >
                      <MessageSquare className="h-3.5 w-3.5 shrink-0 text-violet-400/75" aria-hidden="true" />
                      <span className="truncate">{thread.title}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteThread(thread.id)}
                      aria-label={`Delete thread '${thread.title}'`}
                      className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1.5 rounded-md text-[#8B85A0] hover:text-rose-400 transition-opacity shrink-0"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                    </button>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </div>

      {/* Friendly Privacy Footer */}
      <div className="px-4 py-3 border-t border-white/[0.06] flex items-center gap-2 text-xs text-[#9893A8]">
        <Lock className={`h-3.5 w-3.5 shrink-0 ${shieldEnabled ? 'text-violet-400' : 'text-[#68627C]'}`} aria-hidden="true" />
        <span>{shieldEnabled ? 'Private Mode On' : 'Private Mode Paused'}</span>
      </div>
    </div>
  );

  return (
    <>
      <div className="md:hidden fixed top-0 inset-x-0 z-30 flex items-center justify-between px-4 py-3 bg-[#0B0911]/95 backdrop-blur border-b border-white/[0.08]">
        <button
          type="button"
          onClick={() => onToggleMobile(true)}
          aria-label="Open navigation menu"
          className="p-2 -ml-2 rounded-lg text-[#C4BFE0] hover:text-white"
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="font-display text-base font-semibold text-white whitespace-nowrap shrink-0">
          codexamor
        </span>
        <button
          type="button"
          onClick={onNewThread}
          aria-label="Start new thread"
          className="p-2 -mr-2 rounded-lg text-violet-300"
        >
          <Plus className="h-5 w-5" />
        </button>
      </div>

      <aside className="hidden md:block w-60 shrink-0 h-screen sticky top-0">
        {sidebarContent}
      </aside>

      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40 flex">
          <div
            className="fixed inset-0 bg-black/70"
            onClick={() => onToggleMobile(false)}
            aria-hidden="true"
          />
          <aside className="relative z-50 w-64 max-w-[82vw] h-full">{sidebarContent}</aside>
        </div>
      )}
    </>
  );
}

import { useState, type FormEvent } from 'react';
import { Play, Plus } from 'lucide-react';
import type { FLNode, FLRound, ModelArchitecture } from '../lib/types';

interface FederatedHubViewProps {
  nodes: FLNode[];
  rounds: FLRound[];
  onRegisterNode: (payload: {
    name: string;
    organization: string;
    domain: string;
    architecture: ModelArchitecture;
    sampleCount: number;
  }) => Promise<void>;
  onRunRound: (payload: {
    simulatedSwarmSize: number;
    architecture: ModelArchitecture;
    aggregationStrategy: 'FedAvg' | 'FedProx';
    epsilonPerRound: number;
    clipNorm: number;
  }) => Promise<void>;
}

const PRIVACY_LEVELS: Array<{ id: 'max' | 'balanced' | 'fast'; label: string; epsilon: number }> = [
  { id: 'max', label: 'Maximum Privacy', epsilon: 0.6 },
  { id: 'balanced', label: 'Balanced', epsilon: 1.2 },
  { id: 'fast', label: 'Fast Learning', epsilon: 2.2 },
];

function AccuracyChart({ rounds }: { rounds: FLRound[] }) {
  if (rounds.length === 0) {
    return (
      <div className="h-44 flex items-center justify-center text-sm text-[#8B85A0]">
        Run a community update round to track accuracy over time.
      </div>
    );
  }

  const width = 560;
  const height = 180;
  const padX = 40;
  const padY = 20;
  const innerW = width - padX * 2;
  const innerH = height - padY * 2;

  const points = rounds.map((r, idx) => {
    const x = rounds.length === 1 ? padX + innerW / 2 : padX + (idx / (rounds.length - 1)) * innerW;
    const accY = padY + (1 - Math.min(1, Math.max(0, r.globalAccuracy))) * innerH;
    return { x, accY, r };
  });

  const accPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.accY.toFixed(1)}`).join(' ');

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-44 min-w-[340px]"
        role="img"
        aria-label="Community model accuracy progress"
      >
        {[0.5, 0.7, 0.9, 1.0].map((tick) => {
          const y = padY + (1 - tick) * innerH;
          return (
            <g key={tick}>
              <line x1={padX} y1={y} x2={width - padX} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <text x={6} y={y + 4} fill="#78728F" fontSize="10" fontFamily="DM Sans, sans-serif">
                {Math.round(tick * 100)}%
              </text>
            </g>
          );
        })}

        {rounds.length > 1 && (
          <path d={accPath} fill="none" stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        )}

        {points.map((p) => (
          <g key={p.r.roundNumber}>
            <circle cx={p.x} cy={p.accY} r="4.5" fill="#8B5CF6" stroke="#08070B" strokeWidth="1.5" />
            <text
              x={p.x}
              y={height - 4}
              textAnchor="middle"
              fill="#8B85A0"
              fontSize="10"
              fontFamily="DM Sans, sans-serif"
            >
              Round {p.r.roundNumber}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function FederatedHubView({
  nodes,
  rounds,
  onRegisterNode,
  onRunRound,
}: FederatedHubViewProps) {
  const [swarmSize, setSwarmSize] = useState(250);
  const [privacyLevel, setPrivacyLevel] = useState<'max' | 'balanced' | 'fast'>('balanced');
  const [runningRound, setRunningRound] = useState(false);

  const [showAddNode, setShowAddNode] = useState(false);
  const [nodeName, setNodeName] = useState('');
  const [nodeOrg, setNodeOrg] = useState('');
  const [nodeDomain, setNodeDomain] = useState('General Knowledge');
  const [nodeFormError, setNodeFormError] = useState<string | null>(null);
  const [addingNode, setAddingNode] = useState(false);

  const latestRound = rounds[rounds.length - 1];
  const selectedPrivacy = PRIVACY_LEVELS.find((p) => p.id === privacyLevel) ?? PRIVACY_LEVELS[1];

  const handleTriggerRound = async () => {
    setRunningRound(true);
    try {
      await onRunRound({
        simulatedSwarmSize: swarmSize,
        architecture: 'SMALL_LSTM_GRU',
        aggregationStrategy: 'FedAvg',
        epsilonPerRound: selectedPrivacy.epsilon,
        clipNorm: 1.0,
      });
    } finally {
      setRunningRound(false);
    }
  };

  const handleAddNodeSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!nodeName.trim() || !nodeOrg.trim()) {
      setNodeFormError('Please enter both your name and organization.');
      return;
    }
    setNodeFormError(null);
    setAddingNode(true);
    try {
      await onRegisterNode({
        name: nodeName.trim(),
        organization: nodeOrg.trim(),
        domain: nodeDomain.trim() || 'General Knowledge',
        architecture: 'ANN_MLP',
        sampleCount: 750,
      });
      setNodeName('');
      setNodeOrg('');
      setShowAddNode(false);
    } catch (err) {
      setNodeFormError((err as Error).message);
    } finally {
      setAddingNode(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-20 md:pt-8 pb-16 space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08] pb-5">
        <div>
          <h1 className="font-display text-2xl font-semibold text-white tracking-tight">
            Community Network
          </h1>
          <p className="mt-1 text-sm text-[#9893A8]">
            Improve search intelligence together across public contributors—your personal files never leave your device.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowAddNode((v) => !v)}
          className="inline-flex items-center gap-2 rounded-lg bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.1] px-4 py-2 text-xs font-medium text-white transition-colors whitespace-nowrap shrink-0"
        >
          <Plus className="h-4 w-4 text-violet-400" aria-hidden="true" />
          <span>Join Network</span>
        </button>
      </div>

      {/* Join Network Form */}
      {showAddNode && (
        <form
          onSubmit={handleAddNodeSubmit}
          className="rounded-xl bg-[#120F1B] border border-violet-500/30 p-5 space-y-4"
        >
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold text-white">
              Join as a Public Contributor
            </h2>
            <button
              type="button"
              onClick={() => setShowAddNode(false)}
              className="text-xs text-[#9893A8] hover:text-white"
            >
              Cancel
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label htmlFor="fl-node-name" className="block text-xs text-[#A49FB6] mb-1">
                Display Name
              </label>
              <input
                id="fl-node-name"
                type="text"
                value={nodeName}
                onChange={(e) => setNodeName(e.target.value)}
                placeholder="e.g. Pune Study Group"
                className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label htmlFor="fl-node-org" className="block text-xs text-[#A49FB6] mb-1">
                Organization or School
              </label>
              <input
                id="fl-node-org"
                type="text"
                value={nodeOrg}
                onChange={(e) => setNodeOrg(e.target.value)}
                placeholder="e.g. Engineering Lab"
                className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label htmlFor="fl-node-domain" className="block text-xs text-[#A49FB6] mb-1">
                Topic Focus
              </label>
              <input
                id="fl-node-domain"
                type="text"
                value={nodeDomain}
                onChange={(e) => setNodeDomain(e.target.value)}
                className="w-full rounded-lg bg-[#08070B] border border-white/[0.1] px-3 py-2 text-xs text-white focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={addingNode}
              className="rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-4 py-2 text-xs font-medium text-white transition-colors"
            >
              {addingNode ? 'Joining...' : 'Join Community Pool'}
            </button>
          </div>

          {nodeFormError && (
            <p className="text-xs text-rose-400" role="alert">
              {nodeFormError}
            </p>
          )}
        </form>
      )}

      {/* Simple Summary Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-white/[0.08] rounded-xl bg-[#110E19] border border-white/[0.08]">
        <div className="p-4">
          <div className="text-xs text-[#9893A8]">Active Contributors</div>
          <div className="mt-1 font-display text-2xl font-semibold text-white tabular-nums">
            {nodes.length}
          </div>
        </div>
        <div className="p-4">
          <div className="text-xs text-[#9893A8]">Shared Accuracy</div>
          <div className="mt-1 font-display text-2xl font-semibold text-white tabular-nums">
            {latestRound ? `${(latestRound.globalAccuracy * 100).toFixed(1)}%` : '—'}
          </div>
        </div>
        <div className="p-4">
          <div className="text-xs text-[#9893A8]">Privacy Protection</div>
          <div className="mt-1 font-display text-2xl font-semibold text-violet-300">
            Strong
          </div>
        </div>
      </div>

      {/* Run Update + Accuracy Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-5 rounded-xl bg-[#110E19] border border-white/[0.08] p-5 flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <h2 className="font-display text-base font-semibold text-white">
              Improve Shared AI
            </h2>

            <div>
              <label className="block text-xs text-[#9893A8] mb-1.5">
                Community Participants
              </label>
              <div className="grid grid-cols-4 gap-1.5">
                {[nodes.length, 50, 250, 1000].map((size) => (
                  <button
                    key={size}
                    type="button"
                    onClick={() => setSwarmSize(size)}
                    className={`py-2 rounded-lg text-xs tabular-nums border transition-colors ${
                      swarmSize === size
                        ? 'bg-violet-600/25 border-violet-500/40 text-violet-200 font-medium'
                        : 'border-white/[0.08] text-[#9893A8] hover:text-white'
                    }`}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs text-[#9893A8] mb-1.5">
                Privacy Preference
              </label>
              <div className="grid grid-cols-3 gap-1.5">
                {PRIVACY_LEVELS.map((lvl) => (
                  <button
                    key={lvl.id}
                    type="button"
                    onClick={() => setPrivacyLevel(lvl.id)}
                    className={`px-2 py-2 rounded-lg text-xs border transition-colors ${
                      privacyLevel === lvl.id
                        ? 'bg-violet-600/20 border-violet-500/40 text-violet-200 font-medium'
                        : 'border-white/[0.08] text-[#9893A8] hover:text-white'
                    }`}
                  >
                    {lvl.label.split(' ')[0]}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleTriggerRound}
            disabled={runningRound}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-4 py-2.5 text-sm font-medium text-white transition-colors"
          >
            <Play className="h-4 w-4" aria-hidden="true" />
            <span>{runningRound ? 'Updating...' : 'Run Improvement Round'}</span>
          </button>
        </div>

        <div className="lg:col-span-7 rounded-xl bg-[#110E19] border border-white/[0.08] p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-display text-base font-semibold text-white">
              Accuracy Over Time
            </h2>
            <span className="text-xs tabular-nums text-[#9893A8]">
              {rounds.length} rounds completed
            </span>
          </div>
          <AccuracyChart rounds={rounds} />
        </div>
      </div>

      {/* Simple Contributors Table */}
      <div className="rounded-xl bg-[#110E19] border border-white/[0.08] overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
          <h2 className="font-display text-sm font-semibold text-white">
            Community Contributors
          </h2>
          <span className="text-xs text-[#8B85A0] tabular-nums">{nodes.length} active</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-white/[0.06] text-[#8B85A0]">
                <th className="py-2.5 px-5 font-medium">Contributor</th>
                <th className="py-2.5 px-4 font-medium">Organization</th>
                <th className="py-2.5 px-4 font-medium">Topic</th>
                <th className="py-2.5 px-5 font-medium text-right">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {nodes.map((n) => (
                <tr key={n.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 px-5 font-medium text-white whitespace-nowrap">{n.name}</td>
                  <td className="py-2.5 px-4 text-[#9893A8]">{n.organization}</td>
                  <td className="py-2.5 px-4 text-[#9893A8]">{n.domain}</td>
                  <td className="py-2.5 px-5 text-right tabular-nums text-emerald-400">
                    {(n.localAccuracy * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

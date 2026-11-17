import { useState } from 'react';
import { Activity, Check, Code2, Copy, Download, GraduationCap, Layers, Play, Terminal } from 'lucide-react';
import type { ProductEvaluationReport } from '../lib/evaluator';
import { PYTHON_ARTIFACTS } from '../lib/pythonExport';

const TECH_GROUPS = [
  {
    title: '01. Core Machine Learning & Federated Training',
    items: [
      { name: 'Python 3.14', detail: 'Target runtime across VS Code local venvs and Google Colab notebooks.' },
      { name: 'PyTorch', detail: 'Trains both ANN/MLP dense classifiers and compact Small LSTM/GRU sequence models on edge clients.' },
      { name: 'Flower (flwr)', detail: 'Orchestrates scalable public-domain FedAvg and FedProx aggregation across unlimited virtual and physical clients.' },
      { name: 'Differential Privacy', detail: 'Applies L2 gradient norm clipping plus calibrated Gaussian and Laplace noise with composition accounting.' },
    ],
  },
  {
    title: '02. Natural Language Processing & Retrieval',
    items: [
      { name: 'NLP & Text Preprocessing', detail: 'Normalization, tokenization, stopword filtering, stemming, and local regex PII redaction.' },
      { name: 'Scikit-learn & TF-IDF', detail: 'Smooth IDF vectorization, cosine similarity retrieval, and feature importance scoring across PDF, Word, PPT, Excel, Image & text files.' },
      { name: 'Sentence Transformers', detail: 'Dense semantic embeddings (all-MiniLM-L6-v2) fused with sparse TF-IDF for hybrid local RAG.' },
      { name: 'Pandas & NumPy', detail: 'Vectorized matrix operations, client corpus partitioning, and feature analysis dataframes.' },
    ],
  },
  {
    title: '03. Assistant Synthesis, Storage & Evaluation',
    items: [
      { name: 'LLM & Grounded Web RAG', detail: 'Perplexity-style synthesis with Google Search grounding + open-web retrieval fallback (Wikipedia, DuckDuckGo, arXiv).' },
      { name: 'SQLite', detail: 'Persistent local database storing threads, cited sources, federated round telemetry, and uploaded documents.' },
      { name: 'Streamlit', detail: 'Companion Python 3.14 local dashboard alongside this deployed full-stack React + Express web application.' },
      { name: 'Performance & Error Evaluator', detail: 'Dedicated TypeScript (npm run eval) & Python (evaluate_codexamor.py) benchmark scripts checking F1, MRR, DP RMSE, error rates & latency.' },
    ],
  },
];

export function AboutProjectView() {
  const [selectedId, setSelectedId] = useState(PYTHON_ARTIFACTS[0].id);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [imgFailed, setImgFailed] = useState(false);

  const [evalReport, setEvalReport] = useState<ProductEvaluationReport | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const activeArtifact = PYTHON_ARTIFACTS.find((a) => a.id === selectedId) ?? PYTHON_ARTIFACTS[0];

  const handleCopy = async () => {
    await navigator.clipboard.writeText(activeArtifact.content);
    setCopiedId(activeArtifact.id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleRunEvaluation = async () => {
    setEvalLoading(true);
    setEvalError(null);
    try {
      const res = await fetch('/api/evaluate');
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Evaluation failed.');
      setEvalReport(data);
    } catch (err) {
      setEvalError((err as Error).message);
    } finally {
      setEvalLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 pt-20 md:pt-8 pb-16 space-y-10">
      {/* Hero Banner Split */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-6 flex flex-col justify-center space-y-4">
          <div className="flex items-center gap-2 text-xs text-violet-300">
            <GraduationCap className="h-4 w-4" aria-hidden="true" />
            <span>B.Tech Final Year Capstone Project · Team codexamor</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl font-semibold text-white tracking-tight text-balance">
            Privacy-Preserving Federated NLP Assistant
          </h1>
          <p className="text-sm sm:text-[15px] text-[#B8B2CC] leading-relaxed">
            Built by Team <strong className="text-white">codexamor</strong>, this system combines Perplexity-style live web research and cited synthesis with Sherpa.ai-inspired decentralized federated learning. All technical details, mathematical parameters, performance benchmarks, and source bundles are housed strictly in this section so the main user experience stays effortless.
          </p>
          <div className="pt-1">
            <a
              href="/Codexamor_Capstone_Documentation.pdf"
              download="Codexamor_Capstone_Documentation.pdf"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-violet-600 hover:bg-violet-500 text-white transition-colors"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Download Architecture &amp; Technical Report (PDF)</span>
            </a>
          </div>
        </div>

        <div className="lg:col-span-6 rounded-xl overflow-hidden border border-white/[0.08] bg-[#110E19] min-h-[220px]">
          {!imgFailed ? (
            <img
              src="/src/assets/images/federated_privacy_architecture_1794882509427.jpg"
              alt="Isometric visualization of Codexamor decentralized federated learning nodes with differential privacy"
              referrerPolicy="no-referrer"
              onError={() => setImgFailed(true)}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full min-h-[220px] flex flex-col items-center justify-center gap-2 bg-gradient-to-br from-[#161124] to-[#0B0912] p-6 text-center">
              <Layers className="h-8 w-8 text-violet-400" aria-hidden="true" />
              <span className="text-xs text-[#A49FB6]">Decentralized Federated NLP Architecture</span>
            </div>
          )}
        </div>
      </section>

      {/* Complete Technology Stack */}
      <section aria-label="Project technology stack" className="space-y-5">
        <h2 className="font-display text-xl font-semibold text-white">
          System Architecture &amp; Technical Internals
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {TECH_GROUPS.map((group) => (
            <div
              key={group.title}
              className="rounded-xl bg-[#110E19] border border-white/[0.08] p-5 space-y-4"
            >
              <h3 className="font-display text-sm font-semibold text-violet-300">
                {group.title}
              </h3>
              <dl className="space-y-3">
                {group.items.map((item) => (
                  <div key={item.name} className="border-t border-white/[0.06] pt-2.5 first:border-t-0 first:pt-0">
                    <dt className="text-xs font-semibold text-white">{item.name}</dt>
                    <dd className="mt-0.5 text-xs text-[#9893A8] leading-relaxed">{item.detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </section>

      {/* Live Performance, Parameters & Error-Rate Evaluation */}
      <section aria-label="Performance and error-rate evaluation" className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-xl font-semibold text-white flex items-center gap-2">
              <Activity className="h-5 w-5 text-violet-400" aria-hidden="true" />
              <span>Performance, Parameters &amp; Error-Rate Evaluator</span>
            </h2>
            <p className="mt-1 text-xs text-[#9893A8]">
              Run the dedicated evaluation suite directly here or via <code className="font-mono text-violet-300">npm run eval</code> / <code className="font-mono text-violet-300">python evaluate_codexamor.py</code> in VS Code.
            </p>
          </div>

          <button
            type="button"
            onClick={handleRunEvaluation}
            disabled={evalLoading}
            className="inline-flex items-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-4 py-2 text-xs font-medium text-white transition-colors"
          >
            <Play className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{evalLoading ? 'Evaluating...' : 'Run System Evaluation'}</span>
          </button>
        </div>

        {evalError && (
          <p className="text-xs text-rose-400" role="alert">
            {evalError}
          </p>
        )}

        {evalReport && (
          <div className="rounded-xl bg-[#110E19] border border-white/[0.08] p-5 space-y-5">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="rounded-lg bg-[#0B0912] border border-white/[0.06] p-3">
                <div className="text-[11px] text-[#8B85A0]">Overall Score</div>
                <div className="mt-1 font-mono text-lg font-semibold text-emerald-400">
                  {evalReport.overallScorePercent}% ({evalReport.overallStatus})
                </div>
              </div>
              <div className="rounded-lg bg-[#0B0912] border border-white/[0.06] p-3">
                <div className="text-[11px] text-[#8B85A0]">PII Masking F1 / FNR</div>
                <div className="mt-1 font-mono text-sm font-semibold text-white">
                  F1 {evalReport.privacyMetrics.f1Score} · FNR {evalReport.privacyMetrics.falseNegativeRate}
                </div>
              </div>
              <div className="rounded-lg bg-[#0B0912] border border-white/[0.06] p-3">
                <div className="text-[11px] text-[#8B85A0]">RAG P@1 / Error Rate</div>
                <div className="mt-1 font-mono text-sm font-semibold text-white">
                  {evalReport.ragMetrics.precisionAt1} / {(evalReport.ragMetrics.errorRate * 100).toFixed(1)}%
                </div>
              </div>
              <div className="rounded-lg bg-[#0B0912] border border-white/[0.06] p-3">
                <div className="text-[11px] text-[#8B85A0]">DP Clipping / Gaussian RMSE</div>
                <div className="mt-1 font-mono text-sm font-semibold text-violet-300">
                  {(evalReport.differentialPrivacyMetrics.clippingComplianceRate * 100).toFixed(0)}% / {evalReport.differentialPrivacyMetrics.gaussianEmpiricalRmse}
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="border-b border-white/[0.08] text-[#8B85A0]">
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Strategy</th>
                    <th className="py-2 pr-4">Swarm</th>
                    <th className="py-2 pr-4">ε / Round</th>
                    <th className="py-2 pr-4 text-right">Accuracy</th>
                    <th className="py-2 pr-4 text-right">Error Rate</th>
                    <th className="py-2 pr-4 text-right">Noise σ</th>
                    <th className="py-2 text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.05]">
                  {evalReport.federatedBenchmark.map((row, idx) => (
                    <tr key={idx} className="text-[#DEDAE8]">
                      <td className="py-2 pr-4">{row.architecture}</td>
                      <td className="py-2 pr-4">{row.strategy}</td>
                      <td className="py-2 pr-4">{row.swarmSize}</td>
                      <td className="py-2 pr-4">{row.epsilonPerRound.toFixed(1)}</td>
                      <td className="py-2 pr-4 text-right text-emerald-400">{(row.finalAccuracy * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4 text-right text-amber-300">{(row.errorRate * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4 text-right text-violet-300">{row.avgNoiseSigma}</td>
                      <td className="py-2 text-right">{row.durationMs} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* VS Code Python 3.14 & Google Colab Runnable Suite */}
      <section aria-label="VS Code and Google Colab code bundle" className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="font-display text-xl font-semibold text-white flex items-center gap-2">
              <Terminal className="h-5 w-5 text-violet-400" aria-hidden="true" />
              <span>Run Locally in VS Code (Python 3.14) or Google Colab</span>
            </h2>
            <p className="mt-1 text-xs text-[#9893A8]">
              Download the complete Python 3.14 PyTorch + Flower + Streamlit + Evaluator + SQLite source files or export the pre-configured Google Colab notebook.
            </p>
          </div>
        </div>

        {/* Setup Commands */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl bg-[#110E19] border border-white/[0.08] p-4 space-y-2">
            <div className="text-xs font-semibold text-white">VS Code (Python 3.14 &amp; Node Evaluation)</div>
            <pre className="rounded-lg bg-[#08070B] border border-white/[0.06] p-3 text-xs font-mono text-violet-200 overflow-x-auto">
{`python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate_codexamor.py
streamlit run streamlit_app.py`}
            </pre>
          </div>

          <div className="rounded-xl bg-[#110E19] border border-white/[0.08] p-4 space-y-2">
            <div className="text-xs font-semibold text-white">Google Colab (Zero Local Setup)</div>
            <pre className="rounded-lg bg-[#08070B] border border-white/[0.06] p-3 text-xs font-mono text-violet-200 overflow-x-auto">
{`1. Download Codexamor_Colab_Runner.ipynb below
2. Upload to colab.research.google.com
3. Run all cells (PyTorch, Flower, TF-IDF RAG & DP)`}
            </pre>
          </div>
        </div>

        {/* File Selector + Code Viewer */}
        <div className="rounded-xl bg-[#110E19] border border-white/[0.08] overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-white/[0.06] bg-[#0D0A14]">
            <div className="flex items-center gap-1.5 flex-wrap" role="tablist" aria-label="Python source files">
              {PYTHON_ARTIFACTS.map((art) => (
                <button
                  key={art.id}
                  type="button"
                  role="tab"
                  aria-selected={selectedId === art.id}
                  onClick={() => setSelectedId(art.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors whitespace-nowrap shrink-0 ${
                    selectedId === art.id
                      ? 'bg-violet-600/25 text-violet-200 border border-violet-500/40'
                      : 'text-[#9893A8] hover:text-white border border-transparent'
                  }`}
                >
                  {art.filename}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] text-white transition-colors"
              >
                {copiedId === activeArtifact.id ? (
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
                <span>{copiedId === activeArtifact.id ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                type="button"
                onClick={() => handleDownload(activeArtifact.filename, activeArtifact.content)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-violet-600 hover:bg-violet-500 text-white transition-colors"
              >
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Download {activeArtifact.filename}</span>
              </button>
            </div>
          </div>

          <div className="px-4 py-2.5 border-b border-white/[0.05] flex flex-wrap items-center justify-between gap-2 text-xs text-[#9893A8]">
            <span className="flex items-center gap-1.5">
              <Code2 className="h-3.5 w-3.5 text-violet-400" aria-hidden="true" />
              {activeArtifact.title}
            </span>
            <span>{activeArtifact.runtime}</span>
          </div>

          <pre className="p-4 text-xs font-mono text-[#DEDAE8] bg-[#08070B] overflow-x-auto max-h-[420px] leading-relaxed">
            <code>{activeArtifact.content}</code>
          </pre>
        </div>
      </section>
    </div>
  );
}

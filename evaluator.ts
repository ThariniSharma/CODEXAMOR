import { performance } from 'node:perf_hooks';
import {
  clipGradientNorm,
  computePrivacyComposition,
  privatizeVectorGaussian,
  sampleLaplaceNoise,
} from './differentialPrivacy.ts';
import { executeFederatedRound } from './federated.ts';
import { redactPii, restoreLocalPii } from './privacy.ts';
import { rankLocalRagDocuments } from './tfidfRag.ts';
import type { FLNode, LocalRagDocument, ModelArchitecture } from './types.ts';

export interface ProductEvaluationReport {
  timestamp: string;
  overallStatus: 'PASS' | 'WARN';
  overallScorePercent: number;
  privacyMetrics: {
    casesEvaluated: number;
    precision: number;
    recall: number;
    f1Score: number;
    falsePositiveRate: number;
    falseNegativeRate: number;
    restorationSuccessRate: number;
    avgLatencyMs: number;
  };
  ragMetrics: {
    corpusSize: number;
    queriesEvaluated: number;
    precisionAt1: number;
    meanReciprocalRank: number;
    avgTopCosineSimilarity: number;
    errorRate: number;
    avgLatencyMs: number;
  };
  differentialPrivacyMetrics: {
    clippingComplianceRate: number;
    gaussianEmpiricalRmse: number;
    laplaceEmpiricalRmse: number;
    compositionEpsilonAt10Rounds: number;
    compositionStatus: string;
  };
  federatedBenchmark: Array<{
    architecture: ModelArchitecture;
    strategy: 'FedAvg' | 'FedProx';
    swarmSize: number;
    epsilonPerRound: number;
    finalAccuracy: number;
    errorRate: number;
    finalLoss: number;
    avgNoiseSigma: number;
    durationMs: number;
  }>;
}

const LABELED_PRIVACY_SUITE: Array<{
  text: string;
  customTerms: string[];
  expectedOriginals: string[];
}> = [
  {
    text: 'Email alice.smith@stanford.edu about lab server 10.24.8.15.',
    customTerms: [],
    expectedOriginals: ['alice.smith@stanford.edu', '10.24.8.15'],
  },
  {
    text: 'Call +1-415-555-0192 regarding Aadhaar 123-45-6789 and ProjectOrion.',
    customTerms: ['ProjectOrion'],
    expectedOriginals: ['+1-415-555-0192', '123-45-6789', 'ProjectOrion'],
  },
  {
    text: 'Do not expose token sk-abc1234567890xyz12345678 in public logs.',
    customTerms: [],
    expectedOriginals: ['sk-abc1234567890xyz12345678'],
  },
  {
    text: 'Standard NLP sentence with zero sensitive personal identifiers.',
    customTerms: ['ProjectOrion'],
    expectedOriginals: [],
  },
];

const LABELED_RAG_CORPUS: LocalRagDocument[] = [
  {
    id: 'eval-doc-1',
    title: 'Differential Privacy & Gaussian Noise Clipping',
    category: 'Privacy',
    content: 'Gaussian differential privacy clips client gradients to L2 norm C and adds calibrated sigma noise before FedAvg.',
    tokenCount: 18,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'eval-doc-2',
    title: 'Flower Federated NLP Client Architecture',
    category: 'Federated Learning',
    content: 'Flower flwr orchestrates PyTorch Small GRU and ANN MLP neural networks across distributed public research nodes.',
    tokenCount: 16,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'eval-doc-3',
    title: 'TF-IDF Sparse Retrieval & Cosine Ranking',
    category: 'NLP',
    content: 'Scikit-learn TF-IDF vectorization extracts high-weight domain features and ranks local documents using cosine similarity.',
    tokenCount: 15,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'eval-doc-4',
    title: 'SQLite Local Vault Persistence',
    category: 'Storage',
    content: 'Local documents, chat history threads, and round metrics persist safely inside SQLite without cloud leaks.',
    tokenCount: 16,
    createdAt: '2026-01-01T00:00:00Z',
  },
];

const LABELED_RAG_QUERIES: Array<{ query: string; expectedDocId: string }> = [
  { query: 'How does Gaussian clipping protect client gradients?', expectedDocId: 'eval-doc-1' },
  { query: 'Which Flower PyTorch models run on public nodes?', expectedDocId: 'eval-doc-2' },
  { query: 'How does TF-IDF cosine ranking score features?', expectedDocId: 'eval-doc-3' },
  { query: 'Where do local documents and chat threads persist?', expectedDocId: 'eval-doc-4' },
];

export function runProductEvaluation(): ProductEvaluationReport {
  // 1. Privacy / PII Redaction Evaluation
  const t0Privacy = performance.now();
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  let restorationMatches = 0;
  let restorationTotal = 0;

  for (const sample of LABELED_PRIVACY_SUITE) {
    const audit = redactPii(sample.text, sample.customTerms, true, 0.25);
    const detected = new Set(audit.redactions.map((r) => r.original));
    const expected = new Set(sample.expectedOriginals);

    for (const d of detected) {
      if (expected.has(d)) tp += 1;
      else fp += 1;
    }
    for (const e of expected) {
      if (!detected.has(e)) fn += 1;
    }
    if (expected.size === 0 && detected.size === 0) {
      tn += 1;
    }

    if (audit.redactions.length > 0) {
      restorationTotal += 1;
      const restored = restoreLocalPii(audit.sanitizedQuery, audit.redactions);
      if (restored === sample.text) {
        restorationMatches += 1;
      }
    }
  }
  const privacyMs = (performance.now() - t0Privacy) / LABELED_PRIVACY_SUITE.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const falsePositiveRate = fp + tn > 0 ? fp / (fp + tn) : 0;
  const falseNegativeRate = fn + tp > 0 ? fn / (fn + tp) : 0;

  // 2. Local TF-IDF RAG Evaluation
  const t0Rag = performance.now();
  let hitsAt1 = 0;
  let rrSum = 0;
  let topCosSum = 0;

  for (const item of LABELED_RAG_QUERIES) {
    const { hits } = rankLocalRagDocuments(item.query, LABELED_RAG_CORPUS, 3);
    if (hits[0]?.docId === item.expectedDocId) {
      hitsAt1 += 1;
    }
    topCosSum += hits[0]?.cosineSimilarity ?? 0;
    const rankIdx = hits.findIndex((h) => h.docId === item.expectedDocId);
    if (rankIdx >= 0) {
      rrSum += 1 / (rankIdx + 1);
    }
  }
  const ragMs = (performance.now() - t0Rag) / LABELED_RAG_QUERIES.length;
  const precisionAt1 = hitsAt1 / LABELED_RAG_QUERIES.length;
  const meanReciprocalRank = rrSum / LABELED_RAG_QUERIES.length;

  // 3. Differential Privacy Mechanics Evaluation
  const clipVectors = [
    [3, 4],
    [10, 0, 0],
    [0.2, 0.3],
    [2.5, -2.5, 2.5, -2.5],
  ];
  let clipCompliant = 0;
  for (const vec of clipVectors) {
    const { clipped } = clipGradientNorm(vec, 1.0);
    const norm = Math.sqrt(clipped.reduce((s, v) => s + v * v, 0));
    if (norm <= 1.0 + 1e-6) clipCompliant += 1;
  }

  const zeroVec: number[] = new Array<number>(200).fill(0);
  const gauss = privatizeVectorGaussian(zeroVec, 1.0, 1.2, 1e-5);
  const gaussRmse = Math.sqrt(gauss.noisyVector.reduce((s, v) => s + v * v, 0) / zeroVec.length);
  const lapSamples = zeroVec.map(() => sampleLaplaceNoise(1.0, 1.2));
  const lapRmse = Math.sqrt(lapSamples.reduce((s: number, v: number) => s + v * v, 0) / zeroVec.length);
  const comp10 = computePrivacyComposition(10, 1.2, 1e-5, 25.0, 0.5);

  // 4. Federated Learning Parameter & Error Rate Sweep
  const configs: Array<{
    architecture: ModelArchitecture;
    strategy: 'FedAvg' | 'FedProx';
    swarmSize: number;
    epsilonPerRound: number;
  }> = [
    { architecture: 'ANN_MLP', strategy: 'FedAvg', swarmSize: 50, epsilonPerRound: 0.6 },
    { architecture: 'ANN_MLP', strategy: 'FedAvg', swarmSize: 250, epsilonPerRound: 1.2 },
    { architecture: 'SMALL_LSTM_GRU', strategy: 'FedProx', swarmSize: 250, epsilonPerRound: 1.2 },
    { architecture: 'SMALL_LSTM_GRU', strategy: 'FedProx', swarmSize: 1000, epsilonPerRound: 2.0 },
  ];

  const baseNodes: FLNode[] = Array.from({ length: 12 }, (_, idx) => ({
    id: `eval-node-${idx + 1}`,
    name: `Eval Node ${idx + 1}`,
    organization: 'Codexamor Bench',
    domain: 'NLP',
    architecture: 'ANN_MLP',
    sampleCount: 800 + idx * 45,
    epsilon: 0,
    delta: 1e-5,
    localAccuracy: 0.74,
    localLoss: 0.68,
    status: 'active',
    joinedAt: '2026-01-01T00:00:00Z',
  }));

  const federatedBenchmark = configs.map((cfg) => {
    const t0Fl = performance.now();
    let currentNodes = baseNodes.map((n) => ({ ...n }));
    let prevRound = undefined;
    let sigmaSum = 0;

    for (let r = 0; r < 5; r++) {
      const res = executeFederatedRound({
        nodes: currentNodes,
        previousRound: prevRound,
        simulatedSwarmSize: cfg.swarmSize,
        architecture: cfg.architecture,
        aggregationStrategy: cfg.strategy,
        epsilonPerRound: cfg.epsilonPerRound,
        clipNorm: 1.0,
        rng: () => 0.42 + r * 0.05,
      });
      currentNodes = res.updatedNodes;
      prevRound = res.round;
      sigmaSum += res.round.noiseSigma;
    }

    const durationMs = Number((performance.now() - t0Fl).toFixed(2));
    const finalAccuracy = prevRound?.globalAccuracy ?? 0;
    return {
      architecture: cfg.architecture,
      strategy: cfg.strategy,
      swarmSize: cfg.swarmSize,
      epsilonPerRound: cfg.epsilonPerRound,
      finalAccuracy,
      errorRate: Number((1 - finalAccuracy).toFixed(4)),
      finalLoss: prevRound?.globalLoss ?? 1,
      avgNoiseSigma: Number((sigmaSum / 5).toFixed(3)),
      durationMs,
    };
  });

  const maxFlAccuracy = Math.max(...federatedBenchmark.map((b) => b.finalAccuracy));
  const compositeScore = ((f1Score + precisionAt1 + clipCompliant / clipVectors.length + maxFlAccuracy) / 4) * 100;

  return {
    timestamp: new Date().toISOString(),
    overallStatus: f1Score >= 0.9 && precisionAt1 >= 0.75 ? 'PASS' : 'WARN',
    overallScorePercent: Number(compositeScore.toFixed(1)),
    privacyMetrics: {
      casesEvaluated: LABELED_PRIVACY_SUITE.length,
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1Score: Number(f1Score.toFixed(4)),
      falsePositiveRate: Number(falsePositiveRate.toFixed(4)),
      falseNegativeRate: Number(falseNegativeRate.toFixed(4)),
      restorationSuccessRate: restorationTotal > 0 ? Number((restorationMatches / restorationTotal).toFixed(4)) : 1,
      avgLatencyMs: Number(privacyMs.toFixed(3)),
    },
    ragMetrics: {
      corpusSize: LABELED_RAG_CORPUS.length,
      queriesEvaluated: LABELED_RAG_QUERIES.length,
      precisionAt1: Number(precisionAt1.toFixed(4)),
      meanReciprocalRank: Number(meanReciprocalRank.toFixed(4)),
      avgTopCosineSimilarity: Number((topCosSum / LABELED_RAG_QUERIES.length).toFixed(4)),
      errorRate: Number((1 - precisionAt1).toFixed(4)),
      avgLatencyMs: Number(ragMs.toFixed(3)),
    },
    differentialPrivacyMetrics: {
      clippingComplianceRate: Number((clipCompliant / clipVectors.length).toFixed(4)),
      gaussianEmpiricalRmse: Number(gaussRmse.toFixed(4)),
      laplaceEmpiricalRmse: Number(lapRmse.toFixed(4)),
      compositionEpsilonAt10Rounds: comp10.totalEpsilonBasic,
      compositionStatus: comp10.status,
    },
    federatedBenchmark,
  };
}

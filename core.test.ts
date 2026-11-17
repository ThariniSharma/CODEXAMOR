import test from 'node:test';
import assert from 'node:assert/strict';
import { redactPii, restoreLocalPii, preprocessText } from './privacy';
import {
  clipGradientNorm,
  computePrivacyComposition,
  privatizeVectorGaussian,
  sampleLaplaceNoise,
} from './differentialPrivacy';
import { rankLocalRagDocuments } from './tfidfRag';
import { executeFederatedRound } from './federated';
import type { FLNode, LocalRagDocument } from './types';

test('preprocessText normalizes, removes punctuation, and filters stopwords', () => {
  const tokens = preprocessText('The Privacy-Preserving Federated NLP model is fast!');
  assert.deepEqual(tokens, ['privacy-preserving', 'federated', 'nlp', 'model', 'fast']);
});

test('redactPii masks emails, phones, IPs, and custom terms and restores them locally', () => {
  const input = 'Send notes to alice@university.edu or call 415-555-0192 from 10.0.4.12 about ProjectApollo.';
  const audit = redactPii(input, ['ProjectApollo'], true, 0.25);

  assert.equal(audit.redactions.length, 4);
  assert.ok(!audit.sanitizedQuery.includes('alice@university.edu'));
  assert.ok(!audit.sanitizedQuery.includes('415-555-0192'));
  assert.ok(!audit.sanitizedQuery.includes('10.0.4.12'));
  assert.ok(!audit.sanitizedQuery.includes('ProjectApollo'));
  assert.ok(audit.sanitizedQuery.includes('[EMAIL_1]'));
  assert.ok(audit.sanitizedQuery.includes('[PHONE_1]'));

  const assistantReply = 'I have prepared notes for [EMAIL_1] regarding [CUSTOM_1].';
  const restored = restoreLocalPii(assistantReply, audit.redactions);
  assert.equal(restored, 'I have prepared notes for alice@university.edu regarding ProjectApollo.');
});

test('clipGradientNorm enforces L2 norm ceiling', () => {
  const { clipped, wasClipped } = clipGradientNorm([3, 4], 2.5);
  assert.equal(wasClipped, true);
  const norm = Math.sqrt(clipped[0] ** 2 + clipped[1] ** 2);
  assert.ok(Math.abs(norm - 2.5) < 1e-6);
});

test('Laplace and Gaussian DP mechanisms produce bounded deterministic output with seeded rng', () => {
  const noise = sampleLaplaceNoise(1.0, 1.0, () => 0.75);
  assert.ok(Number.isFinite(noise));
  const { noisyVector, sigma } = privatizeVectorGaussian([1.0, -1.0, 0.5], 1.0, 1.5, 1e-5, () => 0.42);
  assert.equal(noisyVector.length, 3);
  assert.ok(sigma > 0);
  const comp = computePrivacyComposition(4, 0.8, 1e-5, 10.0, 0.25);
  assert.equal(comp.totalEpsilonBasic, 3.45);
  assert.equal(comp.status, 'nominal');
});

test('rankLocalRagDocuments ranks relevant document first with TF-IDF feature weights', () => {
  const docs: LocalRagDocument[] = [
    {
      id: 'd1',
      title: 'Differential Privacy in Flower',
      category: 'Privacy',
      content: 'Differential privacy adds calibrated Gaussian noise to clipped federated gradients in Flower.',
      tokenCount: 14,
      createdAt: '2026-01-01T00:00:00Z',
    },
    {
      id: 'd2',
      title: 'Culinary Pasta Guide',
      category: 'Cooking',
      content: 'Boil salted water and simmer fresh basil tomato sauce for twenty minutes.',
      tokenCount: 12,
      createdAt: '2026-01-01T00:00:00Z',
    },
  ];
  const { hits, globalFeatureAnalysis } = rankLocalRagDocuments('Flower Gaussian differential privacy', docs, 2);
  assert.equal(hits[0].docId, 'd1');
  assert.ok(hits[0].cosineSimilarity > hits[1].cosineSimilarity);
  assert.ok(globalFeatureAnalysis.length > 0);
});

test('executeFederatedRound scales cleanly beyond 3 nodes (12+ and 500 swarm size)', () => {
  const nodes: FLNode[] = Array.from({ length: 12 }, (_, idx) => ({
    id: `node-${idx + 1}`,
    name: `Public Node ${idx + 1}`,
    organization: `Lab ${idx + 1}`,
    domain: 'NLP',
    architecture: 'ANN_MLP',
    sampleCount: 400 + idx * 50,
    epsilon: 0,
    delta: 1e-5,
    localAccuracy: 0.72,
    localLoss: 0.78,
    status: 'active',
    joinedAt: '2026-01-01T00:00:00Z',
  }));

  const { round, updatedNodes } = executeFederatedRound({
    nodes,
    simulatedSwarmSize: 500,
    architecture: 'SMALL_LSTM_GRU',
    aggregationStrategy: 'FedAvg',
    epsilonPerRound: 1.0,
    clipNorm: 1.0,
    rng: () => 0.5,
  });

  assert.equal(round.roundNumber, 1);
  assert.equal(round.participatingNodes, 12);
  assert.equal(round.simulatedSwarmSize, 500);
  assert.equal(updatedNodes.length, 12);
  assert.ok(round.globalAccuracy > 0.72);
  assert.ok(round.globalLoss < 0.78);
});

test('runProductEvaluation reports valid privacy F1, RAG MRR, and FL error rates', async () => {
  const { runProductEvaluation } = await import('./evaluator.ts');
  const report = runProductEvaluation();
  assert.equal(report.overallStatus, 'PASS');
  assert.ok(report.privacyMetrics.f1Score >= 0.9);
  assert.ok(report.ragMetrics.precisionAt1 >= 0.75);
  assert.equal(report.differentialPrivacyMetrics.clippingComplianceRate, 1);
  assert.equal(report.federatedBenchmark.length, 4);
});

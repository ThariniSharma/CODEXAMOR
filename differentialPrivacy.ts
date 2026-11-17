/**
 * Differential Privacy primitives for federated gradient clipping, Laplace/Gaussian noise injection,
 * and privacy budget composition accounting.
 */

export interface DpCompositionResult {
  totalEpsilonBasic: number;
  totalEpsilonStrong: number;
  sigma: number;
  remainingBudget: number;
  status: 'nominal' | 'caution' | 'exhausted';
}

/**
 * Clips a vector's L2 norm to `maxNorm` to bound sensitivity per client update.
 */
export function clipGradientNorm(vector: number[], maxNorm: number): { clipped: number[]; originalNorm: number; wasClipped: boolean } {
  if (maxNorm <= 0) {
    throw new Error('maxNorm must be positive');
  }
  const sqSum = vector.reduce((acc, v) => acc + v * v, 0);
  const norm = Math.sqrt(sqSum);
  if (norm <= maxNorm || norm === 0) {
    return { clipped: [...vector], originalNorm: norm, wasClipped: false };
  }
  const scale = maxNorm / norm;
  return {
    clipped: vector.map((v) => v * scale),
    originalNorm: norm,
    wasClipped: true,
  };
}

/**
 * Samples Laplace(0, sensitivity / epsilon) noise using inverse transform sampling.
 */
export function sampleLaplaceNoise(
  sensitivity: number,
  epsilon: number,
  rng: () => number = Math.random,
): number {
  if (epsilon <= 0) throw new Error('epsilon must be positive');
  const scale = sensitivity / epsilon;
  const u = Math.min(0.999999, Math.max(-0.999999, rng() - 0.5));
  return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

/**
 * Computes Gaussian mechanism standard deviation sigma for (epsilon, delta)-DP.
 */
export function computeGaussianSigma(sensitivity: number, epsilon: number, delta = 1e-5): number {
  if (epsilon <= 0) throw new Error('epsilon must be positive');
  if (delta <= 0 || delta >= 1) throw new Error('delta must be in (0, 1)');
  return (sensitivity * Math.sqrt(2 * Math.log(1.25 / delta))) / epsilon;
}

/**
 * Samples standard normal N(0, 1) via Box-Muller transform and scales by sigma.
 */
export function sampleGaussianNoise(sigma: number, rng: () => number = Math.random): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * sigma;
}

/**
 * Perturbs a vector with (epsilon, delta)-DP Gaussian noise after L2 clipping.
 */
export function privatizeVectorGaussian(
  vector: number[],
  maxNorm: number,
  epsilon: number,
  delta = 1e-5,
  rng: () => number = Math.random,
): { noisyVector: number[]; sigma: number; originalNorm: number } {
  const { clipped, originalNorm } = clipGradientNorm(vector, maxNorm);
  const sigma = computeGaussianSigma(maxNorm, epsilon, delta);
  const noisyVector = clipped.map((v) => v + sampleGaussianNoise(sigma, rng));
  return { noisyVector, sigma, originalNorm };
}

/**
 * Computes sequential and advanced composition bounds over `rounds` and evaluates remaining budget.
 */
export function computePrivacyComposition(
  rounds: number,
  epsilonPerRound: number,
  delta = 1e-5,
  maxBudget = 10.0,
  queryEpsilonSpent = 0,
): DpCompositionResult {
  const safeRounds = Math.max(0, rounds);
  const basic = Number((safeRounds * epsilonPerRound + queryEpsilonSpent).toFixed(3));
  const strong =
    safeRounds === 0
      ? Number(queryEpsilonSpent.toFixed(3))
      : Number(
          (
            Math.sqrt(2 * safeRounds * Math.log(1 / delta)) * epsilonPerRound +
            safeRounds * epsilonPerRound * (Math.exp(epsilonPerRound) - 1) +
            queryEpsilonSpent
          ).toFixed(3),
        );

  const effective = Math.min(basic, strong > 0 ? strong : basic);
  const remaining = Math.max(0, Number((maxBudget - effective).toFixed(3)));
  const ratio = effective / maxBudget;
  const status: DpCompositionResult['status'] =
    ratio >= 0.95 ? 'exhausted' : ratio >= 0.7 ? 'caution' : 'nominal';

  return {
    totalEpsilonBasic: basic,
    totalEpsilonStrong: strong,
    sigma: Number(computeGaussianSigma(1.0, Math.max(0.05, epsilonPerRound), delta).toFixed(4)),
    remainingBudget: remaining,
    status,
  };
}

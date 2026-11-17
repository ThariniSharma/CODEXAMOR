import { preprocessText } from './privacy';
import type { FeatureWeight, LocalRagDocument, RagSearchHit } from './types';

function stemToken(token: string): string {
  if (token.length <= 4) return token;
  if (token.endsWith('ing')) return token.slice(0, -3);
  if (token.endsWith('ied')) return token.slice(0, -3) + 'y';
  if (token.endsWith('ed')) return token.slice(0, -2);
  if (token.endsWith('es')) return token.slice(0, -2);
  if (token.endsWith('s') && !token.endsWith('ss')) return token.slice(0, -1);
  return token;
}

export function tokenizeForRag(text: string): string[] {
  return preprocessText(text).map(stemToken);
}

/**
 * Pure TypeScript TF-IDF Vectorizer + Cosine Similarity RAG retrieval + Feature Importance Analysis.
 * Mirrors scikit-learn TfidfVectorizer (smooth IDF: log((1 + N)/(1 + df)) + 1, L2 normalization).
 */
export function rankLocalRagDocuments(
  query: string,
  documents: LocalRagDocument[],
  topK = 3,
): { hits: RagSearchHit[]; globalFeatureAnalysis: FeatureWeight[] } {
  if (!documents.length || !query.trim()) {
    return { hits: [], globalFeatureAnalysis: [] };
  }

  const docTokensList = documents.map((d) => tokenizeForRag(`${d.title} ${d.content}`));
  const queryTokens = tokenizeForRag(query);
  const N = documents.length;

  // Document frequency per term
  const dfMap = new Map<string, number>();
  for (const tokens of docTokensList) {
    const unique = new Set(tokens);
    for (const term of unique) {
      dfMap.set(term, (dfMap.get(term) ?? 0) + 1);
    }
  }

  const idf = (term: string): number => {
    const df = dfMap.get(term) ?? 0;
    return Math.log((1 + N) / (1 + df)) + 1;
  };

  const buildNormalizedTfIdf = (tokens: string[]): Map<string, number> => {
    const tf = new Map<string, number>();
    for (const t of tokens) {
      tf.set(t, (tf.get(t) ?? 0) + 1);
    }
    const vec = new Map<string, number>();
    let sqNorm = 0;
    for (const [term, count] of tf.entries()) {
      const weight = (1 + Math.log(count)) * idf(term);
      vec.set(term, weight);
      sqNorm += weight * weight;
    }
    const norm = Math.sqrt(sqNorm) || 1;
    for (const [term, w] of vec.entries()) {
      vec.set(term, w / norm);
    }
    return vec;
  };

  const queryVec = buildNormalizedTfIdf(queryTokens);

  // Global feature importance across local corpus
  const corpusFeatureMap = new Map<string, number>();
  const hits: RagSearchHit[] = [];

  documents.forEach((doc, idx) => {
    const tokens = docTokensList[idx];
    const docVec = buildNormalizedTfIdf(tokens);

    let cosine = 0;
    for (const [term, qWeight] of queryVec.entries()) {
      const dWeight = docVec.get(term);
      if (dWeight !== undefined) {
        cosine += qWeight * dWeight;
      }
    }

    const docFeatures: FeatureWeight[] = [];
    for (const [term, w] of docVec.entries()) {
      corpusFeatureMap.set(term, (corpusFeatureMap.get(term) ?? 0) + w);
      docFeatures.push({
        term,
        tfidf: Number(w.toFixed(4)),
        df: dfMap.get(term) ?? 1,
      });
    }
    docFeatures.sort((a, b) => b.tfidf - a.tfidf);

    // Extract best matching sentence snippet
    const sentences = doc.content.split(/(?<=[.!?])\s+/).filter(Boolean);
    let bestSentence = doc.content.slice(0, 220);
    let bestOverlap = -1;
    for (const s of sentences) {
      const sTokens = new Set(tokenizeForRag(s));
      let overlap = 0;
      for (const qt of queryTokens) {
        if (sTokens.has(qt)) overlap += 1;
      }
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestSentence = s.slice(0, 260);
      }
    }

    hits.push({
      docId: doc.id,
      title: doc.title,
      category: doc.category,
      excerpt: bestSentence,
      cosineSimilarity: Number(cosine.toFixed(4)),
      topFeatures: docFeatures.slice(0, 6),
    });
  });

  hits.sort((a, b) => b.cosineSimilarity - a.cosineSimilarity);

  const globalFeatureAnalysis: FeatureWeight[] = Array.from(corpusFeatureMap.entries())
    .map(([term, score]) => ({
      term,
      tfidf: Number((score / N).toFixed(4)),
      df: dfMap.get(term) ?? 1,
    }))
    .sort((a, b) => b.tfidf - a.tfidf)
    .slice(0, 12);

  return {
    hits: hits.slice(0, topK),
    globalFeatureAnalysis,
  };
}

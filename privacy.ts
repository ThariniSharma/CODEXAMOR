import type { PiiRedaction, PrivacyAudit } from './types';

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
  'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was', 'were',
  'will', 'with', 'this', 'these', 'those', 'what', 'which', 'who', 'how', 'when',
  'where', 'why', 'can', 'could', 'should', 'would', 'about', 'into', 'over',
]);

/**
 * NLP text preprocessing pipeline: lowercase, punctuation stripping, tokenization, stopword filtering.
 */
export function preprocessText(input: string): string[] {
  const normalized = input
    .toLowerCase()
    .replace(/\[[A-Z_]+_\d+\]/g, ' ')
    .replace(/[^a-z0-9\s-]/g, ' ');

  return normalized
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

interface PatternSpec {
  category: PiiRedaction['category'];
  regex: RegExp;
}

const PII_PATTERNS: PatternSpec[] = [
  {
    category: 'API_KEY',
    regex: /\b(?:sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{20,}|ghp_[A-Za-z0-9]{20,})\b/g,
  },
  {
    category: 'EMAIL',
    regex: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  },
  {
    category: 'CARD',
    regex: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
  },
  {
    category: 'GOV_ID',
    regex: /\b(?:\d{3}-\d{2}-\d{4}|[A-Z]{5}\d{4}[A-Z]|\d{4}\s\d{4}\s\d{4})\b/g,
  },
  {
    category: 'IP_ADDR',
    regex: /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g,
  },
  {
    category: 'PHONE',
    regex: /(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b/g,
  },
];

interface RawMatch {
  start: number;
  end: number;
  original: string;
  category: PiiRedaction['category'];
}

/**
 * Redacts PII and custom sensitive terms locally before any query is sent to a remote LLM or search engine.
 */
export function redactPii(
  input: string,
  customTerms: string[] = [],
  shieldEnabled = true,
  epsilonPerQuery = 0.25,
): PrivacyAudit {
  const preprocessedTokens = preprocessText(input);

  if (!shieldEnabled || !input.trim()) {
    return {
      shieldEnabled,
      sanitizedQuery: input,
      redactions: [],
      epsilonSpent: 0,
      preprocessedTokens,
    };
  }

  const matches: RawMatch[] = [];

  for (const spec of PII_PATTERNS) {
    spec.regex.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = spec.regex.exec(input)) !== null) {
      matches.push({
        start: m.index,
        end: m.index + m[0].length,
        original: m[0],
        category: spec.category,
      });
    }
  }

  for (const rawTerm of customTerms) {
    const term = rawTerm.trim();
    if (term.length < 2) continue;
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const customRegex = new RegExp(`\\b${escaped}\\b`, 'gi');
    let m: RegExpExecArray | null;
    while ((m = customRegex.exec(input)) !== null) {
      matches.push({
        start: m.index,
        end: m.index + m[0].length,
        original: m[0],
        category: 'CUSTOM',
      });
    }
  }

  // Sort by start offset, preferring longer spans on overlap
  matches.sort((a, b) => a.start - b.start || b.end - a.end);

  const nonOverlapping: RawMatch[] = [];
  let cursor = 0;
  for (const match of matches) {
    if (match.start >= cursor) {
      nonOverlapping.push(match);
      cursor = match.end;
    }
  }

  const categoryCounts: Record<string, number> = {};
  const redactions: PiiRedaction[] = [];
  let sanitized = '';
  let lastIndex = 0;

  for (const m of nonOverlapping) {
    categoryCounts[m.category] = (categoryCounts[m.category] ?? 0) + 1;
    const count = categoryCounts[m.category];
    const token = `[${m.category}_${count}]`;
    sanitized += input.slice(lastIndex, m.start) + token;
    redactions.push({
      id: `${m.category}_${count}`,
      token,
      original: m.original,
      category: m.category,
      start: m.start,
      end: m.end,
    });
    lastIndex = m.end;
  }

  sanitized += input.slice(lastIndex);

  return {
    shieldEnabled: true,
    sanitizedQuery: sanitized,
    redactions,
    epsilonSpent: Number((redactions.length > 0 ? epsilonPerQuery : epsilonPerQuery * 0.4).toFixed(3)),
    preprocessedTokens,
  };
}

/**
 * Reverses local redaction tokens strictly inside the user's browser session.
 */
export function restoreLocalPii(text: string, redactions: PiiRedaction[]): string {
  if (!redactions.length) return text;
  let restored = text;
  for (const r of redactions) {
    restored = restored.split(r.token).join(r.original);
  }
  return restored;
}

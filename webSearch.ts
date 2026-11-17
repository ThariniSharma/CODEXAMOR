import type { SourceItem } from './types.ts';

export interface PublicWebSnippet {
  title: string;
  url: string;
  domain: string;
  snippet: string;
}

function stripHtml(raw: string): string {
  return raw.replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&amp;/g, '&').trim();
}

async function fetchWithTimeout(url: string, timeoutMs = 4500): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': 'CodexamorResearchBot/1.0' },
    });
  } finally {
    clearTimeout(timer);
  }
}

async function searchWikipedia(query: string): Promise<PublicWebSnippet[]> {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(
      query,
    )}&srlimit=3&format=json&origin=*`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      query?: { search?: Array<{ title: string; snippet: string }> };
    };
    return (data.query?.search ?? []).map((item) => ({
      title: item.title,
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(item.title.replace(/\s+/g, '_'))}`,
      domain: 'en.wikipedia.org',
      snippet: stripHtml(item.snippet),
    }));
  } catch {
    return [];
  }
}

async function searchDuckDuckGo(query: string): Promise<PublicWebSnippet[]> {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(
      query,
    )}&format=json&no_html=1&skip_disambig=1`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      Heading?: string;
      AbstractText?: string;
      AbstractURL?: string;
      RelatedTopics?: Array<{ Text?: string; FirstURL?: string }>;
    };
    const results: PublicWebSnippet[] = [];
    if (data.AbstractText && data.AbstractURL) {
      results.push({
        title: data.Heading || 'Web Reference',
        url: data.AbstractURL,
        domain: new URL(data.AbstractURL).hostname.replace(/^www\./, ''),
        snippet: data.AbstractText,
      });
    }
    for (const topic of (data.RelatedTopics ?? []).slice(0, 2)) {
      if (topic.Text && topic.FirstURL) {
        results.push({
          title: topic.Text.split(' - ')[0]?.slice(0, 80) || 'Related Topic',
          url: topic.FirstURL,
          domain: new URL(topic.FirstURL).hostname.replace(/^www\./, ''),
          snippet: topic.Text,
        });
      }
    }
    return results;
  } catch {
    return [];
  }
}

async function searchArxiv(query: string): Promise<PublicWebSnippet[]> {
  try {
    const url = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(
      query,
    )}&start=0&max_results=2`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const xml = await res.text();
    const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];
    return entries
      .map((entry) => {
        const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.replace(/\s+/g, ' ').trim();
        const summary = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.replace(/\s+/g, ' ').trim();
        const id = entry.match(/<id>([\s\S]*?)<\/id>/)?.[1]?.trim();
        if (!title || !id) return null;
        return {
          title,
          url: id,
          domain: 'arxiv.org',
          snippet: (summary ?? '').slice(0, 320),
        };
      })
      .filter((x): x is PublicWebSnippet => x !== null);
  } catch {
    return [];
  }
}

/**
 * Fetches live public-web sources from Wikipedia, DuckDuckGo, and arXiv
 * when Google Search Grounding quota is temporarily unavailable.
 */
export async function fetchOpenWebResults(query: string, includeAcademic: boolean): Promise<PublicWebSnippet[]> {
  const tasks = [searchWikipedia(query), searchDuckDuckGo(query)];
  if (includeAcademic) {
    tasks.push(searchArxiv(query));
  }
  const settled = await Promise.all(tasks);
  const merged = settled.flat();

  const dedup = new Map<string, PublicWebSnippet>();
  for (const item of merged) {
    if (item.url && !dedup.has(item.url)) {
      dedup.set(item.url, item);
    }
  }
  return Array.from(dedup.values()).slice(0, 5);
}

export function toSourceItems(snippets: PublicWebSnippet[], startId: number): SourceItem[] {
  return snippets.map((s, idx) => ({
    id: startId + idx,
    title: s.title,
    url: s.url,
    domain: s.domain,
    snippet: s.snippet,
    origin: 'web',
  }));
}

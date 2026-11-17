export type FocusMode = 'all' | 'research' | 'rag' | 'academic' | 'code';

export type ModelArchitecture = 'ANN_MLP' | 'SMALL_LSTM_GRU';

export interface SourceItem {
  id: number;
  title: string;
  url: string;
  domain: string;
  snippet: string;
  origin: 'web' | 'local_rag';
  similarity?: number;
}

export interface PiiRedaction {
  id: string;
  token: string;
  original: string;
  category: 'EMAIL' | 'PHONE' | 'GOV_ID' | 'CARD' | 'IP_ADDR' | 'API_KEY' | 'CUSTOM';
  start: number;
  end: number;
}

export interface PrivacyAudit {
  shieldEnabled: boolean;
  sanitizedQuery: string;
  redactions: PiiRedaction[];
  epsilonSpent: number;
  preprocessedTokens: string[];
}

export interface ChatMessage {
  id: string;
  threadId: string;
  role: 'user' | 'assistant';
  content: string;
  focusMode: FocusMode;
  sources: SourceItem[];
  followUps: string[];
  privacyAudit?: PrivacyAudit;
  createdAt: string;
}

export interface Thread {
  id: string;
  title: string;
  focusMode: FocusMode;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

export interface FLNode {
  id: string;
  name: string;
  organization: string;
  domain: string;
  architecture: ModelArchitecture;
  sampleCount: number;
  epsilon: number;
  delta: number;
  localAccuracy: number;
  localLoss: number;
  status: 'active' | 'syncing' | 'idle';
  joinedAt: string;
}

export interface FLRound {
  roundNumber: number;
  participatingNodes: number;
  simulatedSwarmSize: number;
  architecture: ModelArchitecture;
  aggregationStrategy: 'FedAvg' | 'FedProx';
  epsilonPerRound: number;
  clipNorm: number;
  noiseSigma: number;
  globalAccuracy: number;
  globalLoss: number;
  avgCosineDrift: number;
  createdAt: string;
}

export interface LocalRagDocument {
  id: string;
  title: string;
  category: string;
  content: string;
  tokenCount: number;
  createdAt: string;
}

export interface FeatureWeight {
  term: string;
  tfidf: number;
  df: number;
}

export interface RagSearchHit {
  docId: string;
  title: string;
  category: string;
  excerpt: string;
  cosineSimilarity: number;
  topFeatures: FeatureWeight[];
}

import fs from 'node:fs';
import path from 'node:path';
import initSqlJs, { type Database } from 'sql.js';
import type { ChatMessage, FLNode, FLRound, FocusMode, LocalRagDocument, Thread } from '../lib/types.ts';

function resolveDataDir(): string {
  const localDir = path.resolve(process.cwd(), 'data');
  try {
    fs.mkdirSync(localDir, { recursive: true });
    fs.accessSync(localDir, fs.constants.W_OK);
    return localDir;
  } catch {
    const tmpDir = path.resolve('/tmp', 'codexamor-data');
    fs.mkdirSync(tmpDir, { recursive: true });
    return tmpDir;
  }
}

const DATA_DIR = resolveDataDir();
const DB_FILE = path.join(DATA_DIR, 'codexamor.db');

let dbInstance: Database | null = null;

function persistDb(): void {
  if (!dbInstance) return;
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    const data = dbInstance.export();
    fs.writeFileSync(DB_FILE, Buffer.from(data));
  } catch {
    // Non-fatal in read-only ephemeral environments
  }
}

const SEED_NODES: Omit<FLNode, 'joinedAt'>[] = [
  { id: 'node-mit-nlp', name: 'Public-Node-01 (Cambridge)', organization: 'OpenNLP Lab', domain: 'Academic NLP', architecture: 'ANN_MLP', sampleCount: 1420, epsilon: 1.6, delta: 1e-5, localAccuracy: 0.842, localLoss: 0.431, status: 'active' },
  { id: 'node-aiims-med', name: 'Public-Node-02 (Clinical)', organization: 'MedPrivacy Consortium', domain: 'Healthcare', architecture: 'SMALL_LSTM_GRU', sampleCount: 1180, epsilon: 1.2, delta: 1e-5, localAccuracy: 0.819, localLoss: 0.468, status: 'active' },
  { id: 'node-eth-sec', name: 'Public-Node-03 (Zurich)', organization: 'CryptoML Collective', domain: 'Security', architecture: 'ANN_MLP', sampleCount: 980, epsilon: 1.8, delta: 1e-5, localAccuracy: 0.854, localLoss: 0.402, status: 'active' },
  { id: 'node-iit-btech', name: 'Public-Node-04 (Codexamor)', organization: 'B.Tech Research Cluster', domain: 'Multilingual NLP', architecture: 'SMALL_LSTM_GRU', sampleCount: 1640, epsilon: 1.4, delta: 1e-5, localAccuracy: 0.861, localLoss: 0.389, status: 'active' },
  { id: 'node-tokyo-edge', name: 'Public-Node-05 (Edge-JP)', organization: 'Community Edge Pool', domain: 'General Search', architecture: 'ANN_MLP', sampleCount: 740, epsilon: 1.1, delta: 1e-5, localAccuracy: 0.798, localLoss: 0.512, status: 'active' },
  { id: 'node-berlin-rag', name: 'Public-Node-06 (Berlin)', organization: 'OpenWeb Indexers', domain: 'RAG Retrieval', architecture: 'ANN_MLP', sampleCount: 1310, epsilon: 1.5, delta: 1e-5, localAccuracy: 0.835, localLoss: 0.447, status: 'active' },
  { id: 'node-toronto-dp', name: 'Public-Node-07 (Toronto)', organization: 'Vector Privacy Group', domain: 'Differential Privacy', architecture: 'SMALL_LSTM_GRU', sampleCount: 1550, epsilon: 1.3, delta: 1e-5, localAccuracy: 0.849, localLoss: 0.415, status: 'active' },
  { id: 'node-singapore-fin', name: 'Public-Node-08 (SG-Fin)', organization: 'Open FinTech Guild', domain: 'Financial Compliance', architecture: 'ANN_MLP', sampleCount: 890, epsilon: 1.0, delta: 1e-5, localAccuracy: 0.806, localLoss: 0.491, status: 'active' },
  { id: 'node-paris-law', name: 'Public-Node-09 (Paris)', organization: 'Civic AI Commons', domain: 'Legal & Policy', architecture: 'SMALL_LSTM_GRU', sampleCount: 1120, epsilon: 1.4, delta: 1e-5, localAccuracy: 0.827, localLoss: 0.459, status: 'active' },
  { id: 'node-austin-code', name: 'Public-Node-10 (Austin)', organization: 'OpenCode Federation', domain: 'Code Intelligence', architecture: 'SMALL_LSTM_GRU', sampleCount: 1490, epsilon: 1.7, delta: 1e-5, localAccuracy: 0.858, localLoss: 0.395, status: 'active' },
  { id: 'node-blr-indic', name: 'Public-Node-11 (Bengaluru)', organization: 'IndicNLP Commons', domain: 'Regional NLP', architecture: 'ANN_MLP', sampleCount: 1360, epsilon: 1.3, delta: 1e-5, localAccuracy: 0.831, localLoss: 0.449, status: 'active' },
  { id: 'node-sydney-bio', name: 'Public-Node-12 (Sydney)', organization: 'BioText Public Node', domain: 'Biomedical', architecture: 'SMALL_LSTM_GRU', sampleCount: 920, epsilon: 1.2, delta: 1e-5, localAccuracy: 0.812, localLoss: 0.478, status: 'active' },
];

const SEED_ROUNDS: Omit<FLRound, 'createdAt'>[] = [
  { roundNumber: 1, participatingNodes: 12, simulatedSwarmSize: 50, architecture: 'ANN_MLP', aggregationStrategy: 'FedAvg', epsilonPerRound: 1.2, clipNorm: 1.0, noiseSigma: 4.0369, globalAccuracy: 0.742, globalLoss: 0.684, avgCosineDrift: 0.412 },
  { roundNumber: 2, participatingNodes: 12, simulatedSwarmSize: 100, architecture: 'ANN_MLP', aggregationStrategy: 'FedAvg', epsilonPerRound: 1.2, clipNorm: 1.0, noiseSigma: 4.0369, globalAccuracy: 0.791, globalLoss: 0.562, avgCosineDrift: 0.338 },
  { roundNumber: 3, participatingNodes: 12, simulatedSwarmSize: 250, architecture: 'SMALL_LSTM_GRU', aggregationStrategy: 'FedProx', epsilonPerRound: 1.2, clipNorm: 1.0, noiseSigma: 4.0369, globalAccuracy: 0.838, globalLoss: 0.448, avgCosineDrift: 0.271 },
];

const SEED_DOCS: Omit<LocalRagDocument, 'createdAt'>[] = [
  {
    id: 'doc-fl-architecture',
    title: 'Sherpa-Style Federated NLP & Flower Architecture',
    category: 'Federated Learning',
    content:
      'Federated learning enables decentralized model training across unlimited public-domain participants without raw text leaving local nodes. Using Flower (flwr) FedAvg and FedProx aggregation, client PyTorch MLP and Small GRU weights are clipped to an L2 norm ceiling before Gaussian differential privacy noise is added.',
    tokenCount: 44,
  },
  {
    id: 'doc-dp-rag',
    title: 'Zero-Trust PII Redaction & Local TF-IDF RAG Protocol',
    category: 'Differential Privacy',
    content:
      'Before a prompt reaches a web search or LLM inference endpoint, local NLP preprocessing detects and masks emails, phone numbers, government identifiers, and custom entities. Local TF-IDF cosine similarity and Sentence Transformer embeddings rank private documents inside the browser or local SQLite vault.',
    tokenCount: 45,
  },
  {
    id: 'doc-btech-eval',
    title: 'Codexamor Benchmark: Privacy Budget vs Convergence',
    category: 'Evaluation',
    content:
      'Across 250 public federated clients, a privacy budget of epsilon 1.2 per round with clip norm 1.0 achieves 84% intent classification accuracy on Small GRU networks while preserving strong (epsilon, delta)-differential privacy guarantees.',
    tokenCount: 34,
  },
];

export async function getDb(): Promise<Database> {
  if (dbInstance) return dbInstance;

  const SQL = await initSqlJs();
  fs.mkdirSync(DATA_DIR, { recursive: true });

  if (fs.existsSync(DB_FILE)) {
    const fileBuf = fs.readFileSync(DB_FILE);
    dbInstance = new SQL.Database(fileBuf);
  } else {
    dbInstance = new SQL.Database();
  }

  dbInstance.run(`
    CREATE TABLE IF NOT EXISTS threads (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      focus_mode TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      focus_mode TEXT NOT NULL,
      sources_json TEXT NOT NULL,
      followups_json TEXT NOT NULL,
      privacy_audit_json TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS fl_nodes (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      organization TEXT NOT NULL,
      domain TEXT NOT NULL,
      architecture TEXT NOT NULL,
      sample_count INTEGER NOT NULL,
      epsilon REAL NOT NULL,
      delta REAL NOT NULL,
      local_accuracy REAL NOT NULL,
      local_loss REAL NOT NULL,
      status TEXT NOT NULL,
      joined_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS fl_rounds (
      round_number INTEGER PRIMARY KEY,
      participating_nodes INTEGER NOT NULL,
      simulated_swarm_size INTEGER NOT NULL,
      architecture TEXT NOT NULL,
      aggregation_strategy TEXT NOT NULL,
      epsilon_per_round REAL NOT NULL,
      clip_norm REAL NOT NULL,
      noise_sigma REAL NOT NULL,
      global_accuracy REAL NOT NULL,
      global_loss REAL NOT NULL,
      avg_cosine_drift REAL NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS rag_documents (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      category TEXT NOT NULL,
      content TEXT NOT NULL,
      token_count INTEGER NOT NULL,
      created_at TEXT NOT NULL
    );
  `);

  const nodeCountRes = dbInstance.exec('SELECT COUNT(*) FROM fl_nodes');
  const nodeCount = Number(nodeCountRes[0]?.values[0]?.[0] ?? 0);
  if (nodeCount === 0) {
    const now = new Date().toISOString();
    const stmt = dbInstance.prepare(
      'INSERT INTO fl_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    );
    for (const n of SEED_NODES) {
      stmt.run([
        n.id,
        n.name,
        n.organization,
        n.domain,
        n.architecture,
        n.sampleCount,
        n.epsilon,
        n.delta,
        n.localAccuracy,
        n.localLoss,
        n.status,
        now,
      ]);
    }
    stmt.free();
  }

  const roundCountRes = dbInstance.exec('SELECT COUNT(*) FROM fl_rounds');
  const roundCount = Number(roundCountRes[0]?.values[0]?.[0] ?? 0);
  if (roundCount === 0) {
    const now = new Date().toISOString();
    const stmt = dbInstance.prepare(
      'INSERT INTO fl_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    );
    for (const r of SEED_ROUNDS) {
      stmt.run([
        r.roundNumber,
        r.participatingNodes,
        r.simulatedSwarmSize,
        r.architecture,
        r.aggregationStrategy,
        r.epsilonPerRound,
        r.clipNorm,
        r.noiseSigma,
        r.globalAccuracy,
        r.globalLoss,
        r.avgCosineDrift,
        now,
      ]);
    }
    stmt.free();
  }

  const docCountRes = dbInstance.exec('SELECT COUNT(*) FROM rag_documents');
  const docCount = Number(docCountRes[0]?.values[0]?.[0] ?? 0);
  if (docCount === 0) {
    const now = new Date().toISOString();
    const stmt = dbInstance.prepare('INSERT INTO rag_documents VALUES (?, ?, ?, ?, ?, ?)');
    for (const d of SEED_DOCS) {
      stmt.run([d.id, d.title, d.category, d.content, d.tokenCount, now]);
    }
    stmt.free();
  }

  persistDb();
  return dbInstance;
}

export async function listThreads(): Promise<Thread[]> {
  const db = await getDb();
  const tRes = db.exec('SELECT id, title, focus_mode, created_at, updated_at FROM threads ORDER BY updated_at DESC');
  if (!tRes.length) return [];

  return tRes[0].values.map((row) => {
    const id = String(row[0]);
    const mRes = db.exec(
      'SELECT id, thread_id, role, content, focus_mode, sources_json, followups_json, privacy_audit_json, created_at FROM messages WHERE thread_id = ? ORDER BY created_at ASC',
      [id],
    );
    const messages: ChatMessage[] = (mRes[0]?.values ?? []).map((mr) => ({
      id: String(mr[0]),
      threadId: String(mr[1]),
      role: String(mr[2]) as 'user' | 'assistant',
      content: String(mr[3]),
      focusMode: String(mr[4]) as FocusMode,
      sources: JSON.parse(String(mr[5] || '[]')),
      followUps: JSON.parse(String(mr[6] || '[]')),
      privacyAudit: mr[7] ? JSON.parse(String(mr[7])) : undefined,
      createdAt: String(mr[8]),
    }));

    return {
      id,
      title: String(row[1]),
      focusMode: String(row[2]) as FocusMode,
      createdAt: String(row[3]),
      updatedAt: String(row[4]),
      messages,
    };
  });
}

export async function upsertThread(thread: {
  id: string;
  title: string;
  focusMode: FocusMode;
}): Promise<void> {
  const db = await getDb();
  const now = new Date().toISOString();
  const existing = db.exec('SELECT id FROM threads WHERE id = ?', [thread.id]);
  if (existing.length && existing[0].values.length) {
    db.run('UPDATE threads SET updated_at = ?, focus_mode = ? WHERE id = ?', [
      now,
      thread.focusMode,
      thread.id,
    ]);
  } else {
    db.run('INSERT INTO threads VALUES (?, ?, ?, ?, ?)', [
      thread.id,
      thread.title,
      thread.focusMode,
      now,
      now,
    ]);
  }
  persistDb();
}

export async function insertMessage(msg: ChatMessage): Promise<void> {
  const db = await getDb();
  db.run('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', [
    msg.id,
    msg.threadId,
    msg.role,
    msg.content,
    msg.focusMode,
    JSON.stringify(msg.sources ?? []),
    JSON.stringify(msg.followUps ?? []),
    msg.privacyAudit ? JSON.stringify(msg.privacyAudit) : null,
    msg.createdAt,
  ]);
  db.run('UPDATE threads SET updated_at = ? WHERE id = ?', [msg.createdAt, msg.threadId]);
  persistDb();
}

export async function deleteThreadById(id: string): Promise<void> {
  const db = await getDb();
  db.run('DELETE FROM messages WHERE thread_id = ?', [id]);
  db.run('DELETE FROM threads WHERE id = ?', [id]);
  persistDb();
}

export async function getFlState(): Promise<{ nodes: FLNode[]; rounds: FLRound[] }> {
  const db = await getDb();
  const nRes = db.exec(
    'SELECT id, name, organization, domain, architecture, sample_count, epsilon, delta, local_accuracy, local_loss, status, joined_at FROM fl_nodes ORDER BY joined_at ASC',
  );
  const nodes: FLNode[] = (nRes[0]?.values ?? []).map((r) => ({
    id: String(r[0]),
    name: String(r[1]),
    organization: String(r[2]),
    domain: String(r[3]),
    architecture: String(r[4]) as FLNode['architecture'],
    sampleCount: Number(r[5]),
    epsilon: Number(r[6]),
    delta: Number(r[7]),
    localAccuracy: Number(r[8]),
    localLoss: Number(r[9]),
    status: String(r[10]) as FLNode['status'],
    joinedAt: String(r[11]),
  }));

  const rRes = db.exec(
    'SELECT round_number, participating_nodes, simulated_swarm_size, architecture, aggregation_strategy, epsilon_per_round, clip_norm, noise_sigma, global_accuracy, global_loss, avg_cosine_drift, created_at FROM fl_rounds ORDER BY round_number ASC',
  );
  const rounds: FLRound[] = (rRes[0]?.values ?? []).map((r) => ({
    roundNumber: Number(r[0]),
    participatingNodes: Number(r[1]),
    simulatedSwarmSize: Number(r[2]),
    architecture: String(r[3]) as FLRound['architecture'],
    aggregationStrategy: String(r[4]) as FLRound['aggregationStrategy'],
    epsilonPerRound: Number(r[5]),
    clipNorm: Number(r[6]),
    noiseSigma: Number(r[7]),
    globalAccuracy: Number(r[8]),
    globalLoss: Number(r[9]),
    avgCosineDrift: Number(r[10]),
    createdAt: String(r[11]),
  }));

  return { nodes, rounds };
}

export async function addFlNode(node: FLNode): Promise<void> {
  const db = await getDb();
  db.run('INSERT INTO fl_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
    node.id,
    node.name,
    node.organization,
    node.domain,
    node.architecture,
    node.sampleCount,
    node.epsilon,
    node.delta,
    node.localAccuracy,
    node.localLoss,
    node.status,
    node.joinedAt,
  ]);
  persistDb();
}

export async function saveFlRound(round: FLRound, updatedNodes: FLNode[]): Promise<void> {
  const db = await getDb();
  db.run('INSERT OR REPLACE INTO fl_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
    round.roundNumber,
    round.participatingNodes,
    round.simulatedSwarmSize,
    round.architecture,
    round.aggregationStrategy,
    round.epsilonPerRound,
    round.clipNorm,
    round.noiseSigma,
    round.globalAccuracy,
    round.globalLoss,
    round.avgCosineDrift,
    round.createdAt,
  ]);

  const stmt = db.prepare(
    'UPDATE fl_nodes SET architecture = ?, local_accuracy = ?, local_loss = ?, epsilon = ?, status = ? WHERE id = ?',
  );
  for (const n of updatedNodes) {
    stmt.run([n.architecture, n.localAccuracy, n.localLoss, n.epsilon, n.status, n.id]);
  }
  stmt.free();
  persistDb();
}

export async function listRagDocuments(): Promise<LocalRagDocument[]> {
  const db = await getDb();
  const res = db.exec(
    'SELECT id, title, category, content, token_count, created_at FROM rag_documents ORDER BY created_at DESC',
  );
  return (res[0]?.values ?? []).map((r) => ({
    id: String(r[0]),
    title: String(r[1]),
    category: String(r[2]),
    content: String(r[3]),
    tokenCount: Number(r[4]),
    createdAt: String(r[5]),
  }));
}

export async function insertRagDocument(doc: LocalRagDocument): Promise<void> {
  const db = await getDb();
  db.run('INSERT INTO rag_documents VALUES (?, ?, ?, ?, ?, ?)', [
    doc.id,
    doc.title,
    doc.category,
    doc.content,
    doc.tokenCount,
    doc.createdAt,
  ]);
  persistDb();
}

export async function deleteRagDocument(id: string): Promise<void> {
  const db = await getDb();
  db.run('DELETE FROM rag_documents WHERE id = ?', [id]);
  persistDb();
}

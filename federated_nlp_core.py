"""
Codexamor — Privacy-Preserving Federated NLP Core (Python 3.14)
Team: codexamor | B.Tech Final Year Capstone Project
Includes:
  - Text Preprocessing + Scikit-learn TF-IDF + Feature Importance Analysis
  - Sentence Transformers dense retrieval + Hybrid Local RAG
  - PyTorch ANN/MLP & Small GRU/LSTM intent/topic classifiers
  - Differential Privacy (L2 gradient clipping + Gaussian/Laplace noise)
  - SQLite local persistence
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.environ.get("CODEXAMOR_DB", "codexamor_local.db")

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "GOV_ID": re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|[A-Z]{5}\d{4}[A-Z])\b"),
    "IP_ADDR": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def init_sqlite(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_sanitized TEXT NOT NULL,
            response_text TEXT NOT NULL,
            epsilon_spent REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fl_rounds (
            round_num INTEGER PRIMARY KEY,
            num_clients INTEGER NOT NULL,
            architecture TEXT NOT NULL,
            global_accuracy REAL NOT NULL,
            global_loss REAL NOT NULL,
            epsilon REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def preprocess_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def redact_pii(text: str) -> tuple[str, list[tuple[str, str]]]:
    sanitized = text
    mappings: list[tuple[str, str]] = []
    for label, pattern in PII_PATTERNS.items():
        for idx, match in enumerate(pattern.finditer(sanitized), start=1):
            token = f"[{label}_{idx}]"
            original = match.group(0)
            sanitized = sanitized.replace(original, token, 1)
            mappings.append((token, original))
    return sanitized, mappings


def apply_dp_noise(
    tensor: torch.Tensor, clip_norm: float = 1.0, epsilon: float = 1.2, delta: float = 1e-5
) -> torch.Tensor:
    norm = torch.norm(tensor, p=2).item()
    scale = min(1.0, clip_norm / max(norm, 1e-8))
    clipped = tensor * scale
    sigma = (clip_norm * np.sqrt(2.0 * np.log(1.25 / delta))) / max(epsilon, 1e-4)
    noise = torch.normal(mean=0.0, std=float(sigma), size=clipped.shape, device=clipped.device)
    return clipped + noise


class FederatedMLP(nn.Module):
    """Lightweight ANN/MLP over TF-IDF or SentenceTransformer embeddings."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FederatedSmallGRU(nn.Module):
    """Compact Small GRU/LSTM sequence classifier for edge federated NLP clients."""

    def __init__(self, vocab_size: int = 4096, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        _, hidden = self.gru(emb)
        return self.head(hidden[-1])


@dataclass
class RagMatch:
    title: str
    score: float
    excerpt: str


class HybridLocalRAG:
    """Combines Scikit-learn TF-IDF feature analysis with Sentence Transformers semantic search."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
        self.encoder = SentenceTransformer(embed_model)
        self.docs: list[dict[str, str]] = []
        self.tfidf_matrix = None
        self.dense_matrix: np.ndarray | None = None

    def index_documents(self, docs: Iterable[dict[str, str]]) -> pd.DataFrame:
        self.docs = list(docs)
        corpus = [preprocess_text(f"{d['title']} {d['content']}") for d in self.docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.dense_matrix = self.encoder.encode([d["content"] for d in self.docs], normalize_embeddings=True)

        feature_names = self.vectorizer.get_feature_names_out()
        mean_scores = np.asarray(self.tfidf_matrix.mean(axis=0)).ravel()
        df_features = pd.DataFrame({"feature": feature_names, "mean_tfidf": mean_scores})
        return df_features.sort_values("mean_tfidf", ascending=False).head(15)

    def search(self, query: str, top_k: int = 3) -> list[RagMatch]:
        if not self.docs or self.tfidf_matrix is None or self.dense_matrix is None:
            return []
        q_tfidf = self.vectorizer.transform([preprocess_text(query)])
        q_dense = self.encoder.encode([query], normalize_embeddings=True)
        sparse_sim = cosine_similarity(q_tfidf, self.tfidf_matrix).ravel()
        dense_sim = (q_dense @ self.dense_matrix.T).ravel()
        hybrid_score = 0.45 * sparse_sim + 0.55 * dense_sim
        ranked_idx = np.argsort(hybrid_score)[::-1][:top_k]
        return [
            RagMatch(
                title=self.docs[i]["title"],
                score=float(hybrid_score[i]),
                excerpt=self.docs[i]["content"][:280],
            )
            for i in ranked_idx
        ]

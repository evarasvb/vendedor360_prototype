"""Implementación simple de BM25."""
import math
from typing import List, Dict
from .preprocessing import tokenize


class BM25:
    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.avg_len = sum(len(tokenize(doc)) for doc in documents) / len(documents)
        self.doc_tokens: List[List[str]] = [tokenize(doc) for doc in documents]
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._build()

    def _build(self):
        # Calcula frecuencia de documentos (df) e idf
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
        num_docs = len(self.documents)
        for term, freq in self.df.items():
            # idf con suavizado
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        scores = []
        for tokens in self.doc_tokens:
            doc_len = len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0)
                freq = tf[term]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                score += idf * numerator / denominator
            scores.append(score)
        return scores

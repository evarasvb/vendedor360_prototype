"""Cálculo manual de TF-IDF y similitud coseno."""
from typing import List, Dict
import math
from .preprocessing import tokenize


class TFIDF:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[List[float]] = []
        self._build()

    def _build(self):
        # Construir vocabulario y df
        df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self.vocab = {term: idx for idx, term in enumerate(df.keys())}
        num_docs = len(self.documents)
        # Calcular idf
        for term, freq in df.items():
            self.idf[term] = math.log(num_docs / (1 + freq)) + 1
        # Calcular vectores tf-idf para cada documento
        for tokens in self.doc_tokens:
            vec = [0.0] * len(self.vocab)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, count in tf.items():
                idx = self.vocab[term]
                tf_weight = count / len(tokens)
                vec[idx] = tf_weight * self.idf[term]
            # normalizar vector
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            self.doc_vectors.append(vec)

    def _query_vector(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        tf: Dict[str, int] = {}
        for t in q_tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(self.vocab)
        total_terms = len(q_tokens) if q_tokens else 1
        for term, count in tf.items():
            if term not in self.vocab:
                continue
            idx = self.vocab[term]
            tf_weight = count / total_terms
            idf = self.idf.get(term, 0.0)
            vec[idx] = tf_weight * idf
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def similarity(self, query: str) -> List[float]:
        q_vec = self._query_vector(query)
        scores: List[float] = []
        for doc_vec in self.doc_vectors:
            # coseno
            score = sum(q * d for q, d in zip(q_vec, doc_vec))
            scores.append(score)
        return scores

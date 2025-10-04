"""Módulo que combina búsqueda BM25 y TF-IDF."""
from typing import List, Tuple
import numpy as np
from .bm25 import BM25
from .tfidf import TFIDF


class HybridSearch:
    """
    Clase para combinar BM25 y TF‑IDF mediante fusión lineal o Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.bm25 = BM25(documents)
        self.tfidf = TFIDF(documents)

    def search(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.5,
        use_rrf: bool = False,
        rank_constant: int = 60,
    ) -> List[Tuple[int, float]]:
        bm_scores = np.array(self.bm25.score(query), dtype=float)
        tfidf_scores = np.array(self.tfidf.similarity(query), dtype=float)
        # Normalizar
        if bm_scores.max() > 0:
            bm_norm = bm_scores / bm_scores.max()
        else:
            bm_norm = bm_scores
        if tfidf_scores.max() > 0:
            tfidf_norm = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_norm = tfidf_scores
        if use_rrf:
            # Calcular rangos (0 para mejor)
            bm_ranks = bm_scores.argsort()[::-1]
            tfidf_ranks = tfidf_scores.argsort()[::-1]
            rrf_scores = np.zeros(len(self.documents))
            # Asignar puntuación RRF a cada documento
            # rrf_score(doc) = sum(1/(rank_constant + rank_position))
            positions_bm = {doc: pos for pos, doc in enumerate(bm_ranks)}
            positions_tf = {doc: pos for pos, doc in enumerate(tfidf_ranks)}
            for doc_id in range(len(self.documents)):
                p_bm = positions_bm.get(doc_id, len(self.documents))
                p_tf = positions_tf.get(doc_id, len(self.documents))
                rrf_scores[doc_id] = 1.0 / (rank_constant + p_bm) + 1.0 / (rank_constant + p_tf)
            # Obtener los top-k según rrf_scores
            top_indices = rrf_scores.argsort()[::-1][:k]
            return [(int(i), float(rrf_scores[i])) for i in top_indices]
        else:
            # Fusión lineal
            final_scores = alpha * bm_norm + (1 - alpha) * tfidf_norm
            top_indices = final_scores.argsort()[::-1][:k]
            return [(int(i), float(final_scores[i])) for i in top_indices]

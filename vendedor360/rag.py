"""Módulo simple de generación de respuestas RAG."""
from typing import List, Tuple


def generate_answer(query: str, docs: List[str], doc_ids: List[str], top_indices: List[int]) -> str:
    """
    Genera una respuesta simple concatenando la primera frase de cada documento recuperado.
    Incluye el identificador de documento como cita.
    """
    parts: List[str] = []
    for idx in top_indices:
        doc_text = docs[idx]
        doc_id = doc_ids[idx]
        # Obtener primera frase (hasta el primer punto)
        sentence = doc_text.strip().split('.')
        summary = sentence[0].strip() if sentence else doc_text.strip()
        parts.append(f"Fuente {doc_id}: {summary}.")
    respuesta = "\n".join(parts)
    return respuesta

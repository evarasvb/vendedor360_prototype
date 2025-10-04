"""Funciones de preprocesamiento para tokenizar texto."""
import re
from typing import List


def tokenize(text: str) -> List[str]:
    """Normaliza y tokeniza una cadena.

    Convierte a minúsculas, reemplaza caracteres no alfanuméricos por espacios y
    divide por espacios en blanco. Retorna una lista de tokens.
    """
    text = text.lower()
    # Reemplaza caracteres que no sean letras, números o acentos por espacio
    text = re.sub(r"[^a-záéíóúñ0-9]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

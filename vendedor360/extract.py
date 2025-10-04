"""Extracción simple de campos de las licitaciones."""
import re
from typing import Dict


def extract_fields(text: str) -> Dict[str, str]:
    """
    Extrae campos clave de un texto de licitación.
    Retorna un diccionario con las claves:
      - plazo_dias (int)
      - garantia (str)
      - anexos (str)
    Si no se encuentra un campo, no se incluye.
    """
    result: Dict[str, str] = {}
    # Buscar plazo en días
    plazo_match = re.search(r"plazo\s+de\s+(\d+)\s*d[ií]as", text.lower())
    if not plazo_match:
        plazo_match = re.search(r"plazo\s+(\d+)\s*d[ií]as", text.lower())
    if plazo_match:
        result["plazo_dias"] = int(plazo_match.group(1))
    # Buscar garantía (porcentaje o años)
    garantia_match = re.search(r"garant[\u00edi]a[^\d]*(\d+%?)", text.lower())
    if garantia_match:
        result["garantia"] = garantia_match.group(1)
    else:
        # Otra forma: número de años
        gar_year = re.search(r"garant[\u00edi]a[^\d]*(\d+)\s*(años|ano|años)", text.lower())
        if gar_year:
            result["garantia"] = gar_year.group(1) + " años"
    # Buscar anexos
    anexo_match = re.search(r"anexo\s*(\d+)", text.lower())
    if anexo_match:
        result["anexos"] = "Anexo " + anexo_match.group(1)
    return result

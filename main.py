"""Script principal para probar el prototipo de Vendedor 360."""
import json
from typing import List
from vendedor360.search import HybridSearch
from vendedor360.extract import extract_fields
from vendedor360.rag import generate_answer



def load_documents(path: str) -> List[dict]:
    """
    Carga un archivo JSON y devuelve una lista de objetos.

    Si ``path`` es relativo, se interpreta relativo a la ubicación de este archivo
    ``main.py`` para que el script funcione independientemente del directorio
    actual. Retorna la lista de diccionarios cargados.
    """
    import os
    # Si la ruta es relativa, conviértela a absoluta usando la ubicación de este archivo
    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



def main():
    # Cargar documentos de ejemplo
    # Utiliza ruta relativa para compatibilidad cuando se ejecuta en otro directorio
    import os
    data = load_documents(os.path.join('datasets', 'sample_data.json'))
    documents = [item['text'] for item in data]
    ids = [item['id'] for item in data]
    # Construir índice híbrido
    searcher = HybridSearch(documents)
    # Solicitar consulta al usuario
    query = input('Ingrese su consulta: ').strip()
    # Buscar
    results = searcher.search(query, k=3, alpha=0.5, use_rrf=False)
    print('\nDocumentos recuperados (índice, puntuación):')
    for idx, score in results:
        print(f" {idx} ({ids[idx]}): {score:.3f}")
        # Mostrar un fragmento del texto
        print('  Texto:', documents[idx][:120] + '...')
        # Extraer campos
        campos = extract_fields(documents[idx])
        print('  Campos extraídos:', campos)
    # Generar respuesta RAG
    top_indices = [idx for idx, _ in results]
    answer = generate_answer(query, documents, ids, top_indices)
    print('\nRespuesta generada:')
    print(answer)


if __name__ == '__main__':
    main()

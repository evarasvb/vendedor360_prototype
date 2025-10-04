# Vendedor 360 (Demostración sin dependencias externas)

Este proyecto es una implementación básica de la arquitectura **Vendedor 360** descrita en el informe adjunto. Debido a que el entorno donde se ejecuta este proyecto no permite instalar bibliotecas de terceros ni acceder a APIs externas, se proporciona un **prototipo mínimo** que ilustra los componentes clave utilizando únicamente la biblioteca estándar de Python.

## Componentes

* **Indexación**: se cargan documentos de ejemplo desde `datasets/sample_data.json` y se genera un índice mixto:
  * **BM25** implementado desde cero para búsqueda léxica.
  * **TF‑IDF** calculado manualmente como representación "densa" y se utiliza el coseno para medir similitud.
  * Se fusionan los resultados mediante una combinación lineal o por **Reciprocal Rank Fusion (RRF)**.

* **Extracción estructurada**: el módulo `extract.py` utiliza expresiones regulares simples para extraer campos como `plazo_dias`, `garantia` y `anexos` de los textos.

* **RAG** (Retrieval‑Augmented Generation): el módulo `rag.py` genera una respuesta concatenando fragmentos relevantes de los documentos recuperados. Debido a la ausencia de un modelo de lenguaje generativo, se devuelve simplemente la primera frase de cada documento como resumen.

* **Ejemplo de uso**: `main.py` carga los datos de ejemplo, construye los índices y permite introducir una consulta desde línea de comandos. A continuación muestra los documentos recuperados, los campos extraídos y la respuesta generada con RAG.

## Ejecución

Para ejecutar la demostración:

```bash
python main.py
```

Se te pedirá que ingreses una consulta (en español). El script devolverá los documentos que mejor concuerdan con la consulta, mostrará los campos extraídos y generará una respuesta con fragmentos citados.

## Limitaciones

* Este prototipo **no usa modelos de aprendizaje profundo** ni APIs externas; por tanto, la "búsqueda semántica" se aproxima con TF‑IDF y la generación de respuestas es muy simple.
* Para un entorno de producción se recomienda utilizar motores de búsqueda como Elasticsearch o OpenSearch, modelos de embeddings reales (`sentence-transformers`, `E5`, etc.) y modelos de generación (GPT‑4o u otros) con soporte de *Structured Outputs* para la extracción de campos.

# ERCOT Projects RAG Chatbot

Sistema de chatbot RAG (Retrieval-Augmented Generation) para responder preguntas sobre proyectos de energía de ERCOT, utilizando búsqueda vectorial, LLMs y análisis de tópicos.

## 🌟 Características Principales

### Sistema RAG Avanzado
- **Dos modos de operación**:
  - **Flash Mode**: Rápido, 2-4 llamadas al LLM con procesamiento directo
  - **Thinking Mode**: Profundo, 5-10 llamadas con validación y expansión de consultas
- **Domain guardrail**: Rechaza preguntas fuera del contexto de ERCOT automáticamente
- **Multi-query retrieval**: Expande consultas para mejor cobertura de documentos
- **Response validation**: Verifica coherencia y formato de respuestas

### Soporte Multilingüe
- **Detección automática de idioma**: Español e Inglés
- **Respuestas en el idioma de la pregunta**
- **Sugerencias de tópicos bilingües**: Las recomendaciones se generan en el idioma detectado
- **Traducción inteligente**: Las consultas en español se traducen para matching de tópicos (modelo entrenado en inglés)


### Interfaz Streamlit
- **Chat interactivo**: Historial de conversación con contexto
- **Visualización de fuentes**: Botones interactivos para ver documentos completos
- **Logging interno**: Procesamiento visible en modo verbose
- **Multi-thread safe**: Manejo correcto de callbacks en threading

### Gestión de Documentos
- **Carga de documentos**: Soporta PDF, TXT, MD
- **Chunking inteligente**: Procesamiento con metadata enriquecida
- **ChromaDB**: Almacenamiento vectorial persistente con embeddings
- **Auto-indexing**: Actualización automática del retriever

```

## 🚀 Inicio Rápido

Ver [SETUP.md](SETUP.md) para instrucciones detalladas de instalación y configuración.

## 🎯 Uso

### Interfaz Web (Streamlit)

```bash
streamlit run frontend.py
```

Abre `http://localhost:8501` en tu navegador.

**Características de la interfaz:**
- 📝 Input de chat con streaming de respuestas
- 📚 Fuentes clicables para ver documentos completos
- 🧠 Sugerencias de tópicos y preguntas de seguimiento
- ⚙️ Configuración de modo (Flash/Thinking), k-docs, auto-summarization
- 📤 Carga de documentos nuevos con indexación automática

### CLI (Terminal)

```bash
python main.py
```

Útil para testing rápido sin interfaz gráfica.

## 🔧 Configuración Avanzada

### Modos RAG

**Flash Mode** (por defecto):
- Rápido y eficiente
- Ideal para preguntas simples
- 2-4 llamadas al LLM

**Thinking Mode**:
- Análisis profundo
- Query expansion + multi-retrieval
- Validación de respuestas
- 5-10 llamadas al LLM

### Variables de Entorno

Ver [SETUP.md](SETUP.md) para la lista completa de variables configurables.


## 🧪 Evaluación

```bash
python -m src.evaluator
```

Evalúa el sistema RAG con métricas de:
- Relevancia de documentos recuperados
- Calidad de respuestas generadas
- Tiempo de respuesta

## 🛠️ Desarrollo

### Agregar nuevos prompts

Edita `src/rag_advanced/prompts.py` para añadir o modificar prompts del sistema.

### Modificar retrieval

Ajusta `src/vector_store.py` para cambiar estrategias de búsqueda o embeddings.

### Personalizar componentes RAG

Los componentes modulares están en `src/rag_advanced/components.py`:
- `is_domain_relevant()`: Domain guardrail
- `classify_question()`: Clasificación de tipo de pregunta
- `validate_response()`: Validación de respuestas
- `expand_query()`: Expansión de consultas
- `extract_query_metadata()`: Extracción de filtros

---

## ➕ Funcionalidades adicionales (opcionales)

### 📤 Add documents (indexación incremental)

Permite **subir nuevos documentos (PDF/TXT/MD) desde la interfaz Streamlit** y **añadirlos incrementalmente** al índice vectorial (ChromaDB), sin reconstruir toda la base de datos.

**Qué hace:**
- Extrae texto del documento subido.
- Genera chunks con el chunker del proyecto (o el pipeline de ingest configurado).
- Inserta los chunks en ChromaDB con metadata para trazabilidad.
- Refresca el retriever para que los documentos nuevos se usen inmediatamente.

**Uso (Streamlit):**
1. Ir a la barra lateral → **Add documents**
2. Seleccionar uno o varios ficheros
3. Pulsar **Index documents**
4. Los documentos quedan disponibles en *Sources / Fuentes* al hacer preguntas
5. Existe la posibilidad de eliminar los documentos en caso de necesidad

---

### 🧠 Topic modeling (BERTopic) para sugerencias

Añade un sistema de **sugerencia de tópicos y preguntas de seguimiento** a partir de:
- **Query topics**: tópicos inferidos desde la query del usuario
- **Chunk topics**: tópicos inferidos desde los chunks recuperados por el RAG (top-k)

El objetivo es **guiar al usuario** hacia preguntas relacionadas y mejorar la exploración del corpus.

#### Características del sistema de tópicos

- **Traducción automática**: Queries en español se traducen al inglés para matching
- **Dual-source topics**: Combina tópicos de la query (intent) con tópicos de documentos recuperados (grounded)
- **Preguntas multilingües**: Templates en español e inglés
- **Limpieza de keywords**: Filtra números, tokens cortos y stop words

#### Entrenar el modelo BERTopic

```bash
python train_topics.py
```

Esto creará `output/bertopic_model.pkl` entrenado con los chunks de ChromaDB.

Para garantizar el correcto funcionamiento del sistema RAG y evitar tiempos elevados de entrenamiento o dependencias de hardware, se proporciona un modelo BERTopic ya entrenado. Puede descargarse desde el siguiente repositorio de Google Drive:

🔗 https://drive.google.com/drive/u/0/folders/1MBH5Ea-6Pq-HkRDi1XMWAdTQC-xD8oqV

Una vez descargado, el archivo bertopic_model.pkl debe colocarse manualmente en la carpeta `output/`



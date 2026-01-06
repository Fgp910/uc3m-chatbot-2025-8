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

## 📊 Análisis de Tópicos

### Entrenar el modelo BERTopic

```bash
python train_topics.py
```

Esto creará `output/bertopic_model.pkl` entrenado con los chunks de ChromaDB.

### Características del sistema de tópicos

- **Traducción automática**: Queries en español se traducen al inglés para matching
- **Dual-source topics**: Combina tópicos de la query (intent) con tópicos de documentos recuperados (grounded)
- **Preguntas multilingües**: Templates en español e inglés
- **Limpieza de keywords**: Filtra números, tokens cortos y stop words

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



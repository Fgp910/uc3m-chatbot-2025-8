# Setup

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure environment

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```
LLM_API_KEY=your_actual_api_key
CHROMADB_PATH=./output/chromadb
```

## 3. Add ChromaDB data

Copy the `chromadb` folder from Person B's pipeline to `./output/chromadb`

Your folder structure should look like:

```
uc3m-chatbot-2025-8/
├── output/
│   └── chromadb/
│       ├── chroma.sqlite3
│       └── dbb87095-.../
├── src/
├── main.py
└── .env
```

## 4. Run

```bash
python main.py
```

## 5. Run evaluation

```bash
python -m src.evaluator
```

## Configuration options

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_API_KEY | (required) | Your API key |
| CHROMADB_PATH | ./output/chromadb | Path to ChromaDB folder |
| COLLECTION_NAME | sgia_chunks | ChromaDB collection name |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence transformer model |

## Features

- Source citations in every response
- Language detection (responds in same language)
- "No information" for out-of-scope questions
- Auto-summarization extension

# Flashoot Company Chatbot

A production-ready RAG-based chatbot using Gemini, Mistral, Groq, and OpenRouter API fallback for Flashoot company data.

## 📁 Project Structure

```
bot/
├── Dockerfile           # Docker image for Render/backend deployment
├── .dockerignore        # Docker build exclusions
├── Procfile             # Render/Gunicorn start command
├── main.py              # Main chatbot application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
├── .env                 # Your API key (not committed)
├── data.json            # Company data (provided)
└── qdrant_db/           # Local vector database (auto-created)
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add at least one API key:

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```
GEMINI_API_KEYS=your_gemini_key_1,your_gemini_key_2
MISTRAL_API_KEYS=your_mistral_key
GROQ_API_KEYS=your_groq_key
OPENROUTER_API_KEYS=your_openrouter_key
```

### 3. Prepare Data

Ensure `data.json` exists in the `bot/` directory with your company data.

## 📄 Data Format

The `data.json` should contain structured company information. Example structure:

```json
{
  "company": {
    "name": "Flashoot",
    "legal_name": "Konchamkode Private Limited",
    "founded": 2023,
    "industry": ["Creator Economy", "Real-time Content Creation"],
    "headquarters": {
      "city": "Hyderabad",
      "country": "India"
    }
  },
  "services": {
    "consumer_services": ["Instant reel creation", "Event videography"],
    "business_services": ["Corporate shoots", "Brand campaigns"]
  },
  "mobile_apps": {
    "android": {
      "package_name": "com.flashoot.user",
      "downloads": "10K+"
    }
  }
}
```

## 🤖 Running the Chatbot

### First Run (Initializes Database)

```bash
python main.py
```

### Subsequent Runs (Uses Existing Database)

```bash
python main.py
```

## 🌐 HTTP API Mode (for Render + GitHub Pages Frontend)

This project supports API mode so you can host the backend on Render and the frontend on GitHub Pages.

### Start API locally

```bash
# Windows PowerShell
$env:RUN_MODE="api"
python main.py
```

API will run on:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/chat` (POST JSON: `{"message": "..."}`)

### CORS / frontend origin control

In `.env`:

```env
RUN_MODE=api
ALLOWED_ORIGINS=*
```

For production, set strict origins, for example:

```env
ALLOWED_ORIGINS=https://yourusername.github.io,http://127.0.0.1:5500
```

## 🖥️ Frontend (`index.html`)

- Open `index.html` in browser (or host via GitHub Pages)
- Set backend URL in UI:
  - Local: `http://127.0.0.1:8000`
  - Render: `https://your-render-service.onrender.com`
- Click **Save** and start chatting

The selected API base URL is persisted in browser `localStorage`.

## Deploy Backend on Render

Recommended: deploy as a Docker-backed Render Web Service for more consistent production behavior.

The Docker image preloads the default embedding model during build:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This keeps Render startup faster and lowers RAM usage compared with `all-mpnet-base-v2`, which is risky on free-tier instances.

1. Push this repo to GitHub.
2. Create a **Render Web Service** from the repo.
3. Select **Docker** as the runtime/environment.
4. Render will build using the included `Dockerfile`.
5. Set environment variables in Render:
   - `RUN_MODE=api`
   - `ALLOWED_ORIGINS=https://yourusername.github.io`
   - your API key variables (e.g., `GEMINI_API_KEYS`, etc.)
6. Render provides `PORT` automatically; the Docker command uses it.
7. Deploy and verify:
   - `https://your-render-service.onrender.com/health`

Alternative non-Docker Render settings:
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn "main:create_app()" --bind 0.0.0.0:$PORT --workers 1 --timeout 180`
- Or let Render use the included `Procfile`.

## 🐳 Local Docker Testing

Build image:

```bash
docker build -t flashoot-ragbot .
```

Run container using your local `.env` file:

```bash
docker run --env-file .env -p 8000:8000 flashoot-ragbot
```

Then open:
- `http://127.0.0.1:8000/health`
- Use `http://127.0.0.1:8000` as backend URL in `index.html`

## ⚠️ Embedding Model / Qdrant Note

Default embedding model:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

If you previously indexed data with `all-mpnet-base-v2` (`768` dimensions), delete/reset your old Qdrant database before running with MiniLM (`384` dimensions), otherwise vector dimension mismatch errors can happen.

Local reset example:

```powershell
Remove-Item -Recurse -Force qdrant_db_v3
```

On Render Docker deploys, the container starts fresh unless you attach persistent disk storage.

## 📄 Deploy Frontend on GitHub Pages

1. Keep `index.html` at repo root (already added).
2. In GitHub repo settings, enable Pages for your branch/root.
3. Open the Pages URL.
4. In the frontend input, paste Render backend URL and click **Save**.

## 💬 Usage

After starting the chatbot:

```
You: what services do you provide?
Bot: Flashoot offers Instant reel creation and Event videography...

You: where are you located?
Bot: Flashoot is headquartered in Hyderabad, India...

You: exit
Bot: Goodbye! 👋
```

## 🛠️ Technical Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` by default for faster Render startup and lower RAM usage
- **Vector Database**: Qdrant (local storage)
- **LLM**: Gemini, Mistral, Groq, and OpenRouter via OpenAI-compatible APIs
- **Chunking**: `langchain-text-splitters` recursive character text splitter
- **Retrieval**: Cosine similarity search (top 5 chunks)
- **Anti-hallucination**: Strict context-only answering

## 🔧 Configuration

Environment variables in `.env`:

- `GEMINI_API_KEYS`: Comma-separated Gemini API keys
- `MISTRAL_API_KEYS`: Comma-separated Mistral API keys
- `GROQ_API_KEYS`: Comma-separated Groq API keys
- `OPENROUTER_API_KEYS`: Comma-separated OpenRouter API keys
- `GEMINI_MODEL`, `MISTRAL_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`: Model names for each provider
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`: Optional generation settings
- `QDRANT_COLLECTION` (optional): Custom collection name
- `QDRANT_PATH` (optional): Local vector database path, default `./qdrant_db_v3`
- `EMBEDDING_MODEL` (optional): Custom embedding model
- `EMBEDDING_DIM` (optional): Embedding vector dimension, default `384`

## ⚠️ Error Handling

The chatbot handles:
- Missing API keys
- Invalid JSON data
- Empty retrieval results
- API limits, quota errors, and connection errors with fallback to the next configured API
- Keyboard interrupts

## 📦 Dependencies

- `langchain`, `langchain-core` - LangChain 1.x base packages
- `langchain-text-splitters` - Text chunking
- `langchain-huggingface` - Hugging Face embeddings
- `langchain-qdrant` - Qdrant vector store integration
- `qdrant-client` - Vector database
- `sentence-transformers` - Embeddings
- `python-dotenv` - Environment variables
- `openai` - OpenAI-compatible API client

## 🎯 Features

✅ Automatic JSON data loading
✅ Intelligent nested JSON flattening
✅ Recursive text chunking
✅ Local vector database persistence
✅ Context-aware answers
✅ Anti-hallucination safeguards
✅ Chat history in memory
✅ Graceful error handling
✅ Interactive CLI interface
✅ HTTP API mode for web frontend
✅ GitHub Pages-friendly frontend (`index.html`)
✅ Render-ready backend configuration
✅ Docker-ready backend deployment

## 🔒 Security

- API key loaded from `.env` (not hardcoded)
- `.env` should be in `.gitignore`
- No sensitive data stored in vector DB

## 📈 Performance

- Fast local embeddings
- Efficient vector search
- API fallback across providers and keys
- Minimal API calls

---

**Flashoot Company Chatbot** | Powered by multi-provider LLM fallback

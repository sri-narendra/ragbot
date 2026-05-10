import os
import json
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from threading import Lock

import qdrant_client
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from qdrant_client.http.models import Distance, VectorParams
from flask import Flask, jsonify, request

# Load .env before reading module-level configuration defaults.
load_dotenv()

# Constants
DATA_FILE = os.getenv("DATA_FILE", "data.json")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db_v3")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "flashoot_company_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 700
SKIPPED_INDEX_SECTIONS = {"source_file_snapshots"}
UNKNOWN_ANSWER_PATTERNS = (
    "i don't know",
    "i do not know",
    "not sure",
    "cannot find",
    "can't find",
    "no information",
    "not available",
)
CUSTOMER_FRIENDLY_FALLBACK = (
    "Thanks for asking! I don’t have that exact detail in my current knowledge base yet, "
    "but Flashoot focuses on fast, high-quality short-form video creation and creator-driven services. "
    "If you’d like, I can share our services, delivery model, and social links."
)
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

_chatbot_instance = None
_chatbot_lock = Lock()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("flashoot-chatbot")


@dataclass
class LLMProvider:
    """OpenAI-compatible LLM provider configuration."""

    name: str
    api_key: str
    base_url: str
    model: str

class CompanyChatbot:
    """Main chatbot class handling RAG pipeline and multi-provider LLM fallback."""

    def __init__(self):
        """Initialize chatbot with embeddings, vector store, and LLM clients."""
        self._validate_environment()
        self._load_llm_providers()

        # Initialize components
        self.embeddings = self._init_embeddings()
        self.collection_was_created = False
        self.vectorstore = self._init_vectorstore()
        self.llm_clients = self._init_llm_clients()
        self.active_provider_index = 0
        self.overview_context = None
        self.chat_history = []

    def _validate_environment(self) -> None:
        """Validate required files and environment."""
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"Data file '{DATA_FILE}' not found. "
                "Please ensure the JSON data file exists."
            )

        if not os.path.exists('.env'):
            raise FileNotFoundError(
                "'.env' file not found. "
                "Please create it with at least one configured API key."
            )

    def _csv_env(self, key: str) -> List[str]:
        """Read a comma-separated environment variable."""
        value = os.getenv(key, "")
        return [item.strip() for item in value.split(",") if item.strip()]

    def _load_llm_providers(self) -> None:
        """Load configured API providers from environment."""
        load_dotenv()

        provider_templates = [
            {
                "name": "gemini",
                "keys": self._csv_env("GEMINI_API_KEYS"),
                "base_url": os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            },
            {
                "name": "mistral",
                "keys": self._csv_env("MISTRAL_API_KEYS"),
                "base_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            },
            {
                "name": "groq",
                "keys": self._csv_env("GROQ_API_KEYS"),
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            },
            {
                "name": "openrouter",
                "keys": self._csv_env("OPENROUTER_API_KEYS"),
                "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            },
        ]

        self.llm_providers = [
            LLMProvider(
                name=provider["name"],
                api_key=api_key,
                base_url=provider["base_url"],
                model=provider["model"],
            )
            for provider in provider_templates
            for api_key in provider["keys"]
        ]

        if not self.llm_providers:
            raise ValueError(
                "No API keys found. Set at least one of GEMINI_API_KEYS, "
                "MISTRAL_API_KEYS, GROQ_API_KEYS, or OPENROUTER_API_KEYS in .env."
            )

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize sentence transformer embeddings."""
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _init_vectorstore(self) -> QdrantVectorStore:
        """Initialize Qdrant vector store with local persistence."""
        client = qdrant_client.QdrantClient(
            path=QDRANT_PATH,
            prefer_grpc=True
        )

        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            self.collection_was_created = True

        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )

    def _init_llm_clients(self) -> List[OpenAI]:
        """Initialize OpenAI-compatible clients for all configured providers."""
        return [
            OpenAI(api_key=provider.api_key, base_url=provider.base_url)
            for provider in self.llm_providers
        ]

    def _load_json_data(self) -> Dict[str, Any]:
        """Load and validate JSON data file."""
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data:
                raise ValueError("JSON data file is empty")

            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON data: {e}")

    def _flatten_json(self, data: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Recursively flatten nested JSON structure."""
        items = {}

        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self._flatten_json(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(item, (dict, list)):
                    items.update(self._flatten_json(item, new_key, sep=sep))
                else:
                    items[new_key] = item
        else:
            items[parent_key] = data

        return items

    def _format_list(self, values: Any) -> str:
        """Format list-like values for compact context text."""
        if not isinstance(values, list):
            return str(values) if values is not None else ""

        formatted = []
        for value in values:
            if isinstance(value, dict):
                formatted.append(", ".join(f"{k}: {v}" for k, v in self._flatten_json(value).items()))
            else:
                formatted.append(str(value))

        return "; ".join(formatted)

    def _build_company_overview_document(self, data: Dict[str, Any]) -> Document:
        """Create a high-signal overview document for broad company questions."""
        company = data.get("company", {})
        services = data.get("services", {})
        expansion = data.get("expansion", {})
        mobile_apps = data.get("mobile_apps", {})
        platform_features = data.get("platform_features", {})

        lines = [
            "FLASHOOT COMPANY OVERVIEW",
            "",
            f"Name: {company.get('name', 'Flashoot')}",
            f"Legal name: {company.get('legal_name', '')}",
            f"Founded: {company.get('founded', '')}",
            f"Core positioning: {company.get('core_positioning', '')}",
            f"Business model: {company.get('business_model', '')}",
            f"Known for: instant short-form video, Instagram Reel creation, event videography, "
            f"creator marketplace services, and rapid reel delivery.",
            f"Delivery claim: {services.get('delivery_claim', '')}",
            f"Consumer services: {self._format_list(services.get('consumer_services', []))}",
            f"Business services: {self._format_list(services.get('business_services', []))}",
            f"Industry: {self._format_list(company.get('industry', []))}",
            f"Tagline: {company.get('tagline', '')}",
            f"Premium packages: {self._format_list(services.get('premium_packages', []))}",
            f"Event types: {self._format_list(services.get('event_types', []))}",
            f"India cities: {self._format_list(expansion.get('india_cities', []))}",
            f"International presence: {self._format_list(expansion.get('international_presence', []))}",
            f"Customer app: {mobile_apps.get('customer_app', {}).get('name', '')}",
            f"Partner app: {mobile_apps.get('partner_app', {}).get('name', '')}",
            f"Platform features: {self._format_list(platform_features.get('booking_system', []))}; "
            f"{self._format_list(platform_features.get('commerce', []))}; "
            f"{self._format_list(platform_features.get('content_tools', []))}",
            "",
            "Question aliases: What is Flashoot? What does Flashoot do? What is your company known for? "
            "Tell me about your company. What services do you offer?",
        ]

        return Document(
            page_content="\n".join(line for line in lines if line is not None),
            metadata={
                "source": DATA_FILE,
                "section": "company_overview",
                "type": "overview",
            },
        )

    def _get_overview_context(self) -> str:
        """Load a compact company overview for broad company questions."""
        if self.overview_context is None:
            self.overview_context = self._build_company_overview_document(
                self._load_json_data()
            ).page_content

        return self.overview_context

    def _json_to_documents(self, data: Dict[str, Any]) -> List[Document]:
        """Convert JSON data to LangChain documents with metadata."""
        documents = [self._build_company_overview_document(data)]

        # Convert each top-level section to a document
        for key, value in data.items():
            if key in SKIPPED_INDEX_SECTIONS:
                continue

            if isinstance(value, dict):
                # Create document from section
                content = f"{key.upper()}\n\n" + "\n".join(
                    f"{k}: {v}" for k, v in self._flatten_json(value).items()
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": DATA_FILE,
                            "section": key,
                            "type": "company_data"
                        }
                    )
                )
            elif isinstance(value, list):
                # Handle list items
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        content = f"{key.upper()} Item {i+1}\n\n" + "\n".join(
                            f"{k}: {v}" for k, v in self._flatten_json(item).items()
                        )

                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": DATA_FILE,
                                    "section": key,
                                    "item_index": i,
                                    "type": "list_item"
                                }
                            )
                        )
                    else:
                        documents.append(
                            Document(
                                page_content=f"{key.upper()} Item {i+1}\n\nvalue: {item}",
                                metadata={
                                    "source": DATA_FILE,
                                    "section": key,
                                    "item_index": i,
                                    "type": "list_item",
                                },
                            )
                        )

        return documents

    def _create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using recursive text splitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        return text_splitter.split_documents(documents)

    def _retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from the vector store."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        retrieval_question = question
        broad_company_terms = (
            "about",
            "company",
            "comany",
            "known",
            "do",
            "does",
            "service",
            "services",
            "offer",
            "flashoot",
        )

        is_broad_company_question = any(term in question.lower() for term in broad_company_terms)

        if is_broad_company_question:
            retrieval_question = (
                f"{question}\n\n"
                "What is Flashoot? What does Flashoot do? What is Flashoot known for? "
                "Flashoot company overview, services, business model, reel creators, "
                "videography, content creation marketplace, Uber for Reels."
            )

        documents = retriever.invoke(retrieval_question)

        if not documents:
            return self._get_overview_context() if is_broad_company_question else ""

        context = "\n\n---\n\n".join(doc.page_content for doc in documents)
        if is_broad_company_question:
            return f"{self._get_overview_context()}\n\n---\n\n{context}"

        return context

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        """Return True when a provider/key should be skipped and the next tried."""
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True

        if isinstance(error, AuthenticationError):
            return True

        if isinstance(error, APIStatusError):
            if error.status_code in {401, 403, 408, 409, 429, 500, 502, 503, 504}:
                return True

            message = str(error).lower()
            return any(word in message for word in ("quota", "rate", "limit", "capacity"))

        message = str(error).lower()
        return any(word in message for word in ("quota", "rate", "limit", "timeout", "capacity"))

    def _provider_order(self) -> List[int]:
        """Try the active provider first, then cycle through the remaining providers."""
        total = len(self.llm_providers)
        return [(self.active_provider_index + offset) % total for offset in range(total)]

    def _call_llm_with_fallback(self, context: str, question: str) -> str:
        """Call configured LLM providers, falling back when limits or API errors occur."""
        system_prompt = (
            "You are a helpful AI assistant for Flashoot company. "
            "Answer questions based ONLY on the provided context. "
            "If the answer is not present in the context, do NOT say 'I don't know'. "
            "Instead, respond politely with a customer-friendly fallback and offer nearby helpful company info. "
            "Be concise, factual, and professional. "
            "Do not make up information or speculate."
        )
        user_prompt = f"Context:\n{context or 'No relevant context found.'}\n\nQuestion: {question}"
        last_error = None

        for index in self._provider_order():
            provider = self.llm_providers[index]
            client = self.llm_clients[index]

            try:
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
                    max_tokens=int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                )
                self.active_provider_index = index
                content = response.choices[0].message.content.strip()
                return self._normalize_uncertain_response(content)

            except Exception as error:
                last_error = error
                if self._is_retryable_llm_error(error):
                    print(f"\n⚠️ {provider.name} unavailable or limited. Trying next API...")
                    continue
                raise

        raise RuntimeError(f"All configured APIs failed. Last error: {last_error}")

    def _normalize_uncertain_response(self, response: str) -> str:
        """Replace blunt uncertainty phrases with a customer-friendly fallback."""
        normalized = (response or "").strip()
        lowered = normalized.lower()

        if not normalized:
            return CUSTOMER_FRIENDLY_FALLBACK

        if any(pattern in lowered for pattern in UNKNOWN_ANSWER_PATTERNS):
            return CUSTOMER_FRIENDLY_FALLBACK

        return normalized

    def _is_capability_question(self, question: str) -> bool:
        """Detect generic capability questions like 'what can you do'."""
        q = (question or "").strip().lower()
        capability_patterns = (
            "what can you do",
            "what do you do",
            "how can you help",
            "your services",
            "services you offer",
        )
        return any(pattern in q for pattern in capability_patterns)

    def _capability_response(self) -> str:
        """Return a concise, customer-friendly capability response."""
        return (
            "I can help you with Flashoot information like services, pricing packages, "
            "booking flow, delivery model, social links, and company details. "
            "For quick help, ask things like: 'What services do you offer?', "
            "'How fast is delivery?', or 'Share your social media links.'"
        )

    def ask(self, question: str) -> str:
        """Answer a user question using retrieved company context."""
        if self._is_capability_question(question):
            answer = self._capability_response()
            logger.info("Q: %s", question)
            logger.info("A: %s", answer)
            return answer

        context = self._retrieve_context(question)
        answer = self._call_llm_with_fallback(context, question)
        logger.info("Q: %s", question)
        logger.info("A: %s", answer)
        return answer

    def initialize_database(self) -> None:
        """Load data, create chunks, generate embeddings, and store in vector DB."""
        print("🔄 Initializing company knowledge base...")

        try:
            # Load and process data
            data = self._load_json_data()
            documents = self._json_to_documents(data)
            chunks = self._create_chunks(documents)

            print(f"📄 Loaded {len(documents)} documents")
            print(f"📦 Created {len(chunks)} chunks")

            # Store in vector database
            self.vectorstore.add_documents(chunks)

            print(f"💾 Stored embeddings in {QDRANT_PATH}")
            print("✅ Knowledge base initialized successfully!")

        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            raise

    def chat(self) -> None:
        """Start interactive chat session."""
        print("🤖 Flashoot Company Chatbot")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Exit conditions
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: Goodbye! 👋")
                    break

                if not user_input:
                    continue

                # Add to chat history
                self.chat_history.append({"role": "user", "content": user_input})

                # Get response from RAG chain
                print("Bot: ", end="", flush=True)

                try:
                    response = self.ask(user_input)
                    print(response)

                    # Add response to history
                    self.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    print(f"\n❌ Error getting response: {e}")
                    print("Bot: I encountered an error processing your request.")

            except KeyboardInterrupt:
                print("\nBot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Bot: Something went wrong. Let's try again.")

    def close(self) -> None:
        """Close local vector database resources."""
        client = getattr(getattr(self, "vectorstore", None), "client", None)
        if client and hasattr(client, "close"):
            client.close()


def main():
    """Main entry point."""
    chatbot = None
    try:
        # Initialize chatbot
        chatbot = CompanyChatbot()

        # Initialize data when the vector collection was freshly created.
        if chatbot.collection_was_created:
            chatbot.initialize_database()
        else:
            print("💾 Using existing knowledge base...")

        # Start chat
        chatbot.chat()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        if chatbot is not None:
            chatbot.close()


def _get_chatbot() -> CompanyChatbot:
    """Create or return a singleton chatbot instance for API mode."""
    global _chatbot_instance

    with _chatbot_lock:
        if _chatbot_instance is None:
            _chatbot_instance = CompanyChatbot()
            if _chatbot_instance.collection_was_created:
                _chatbot_instance.initialize_database()
        return _chatbot_instance


def _resolve_cors_origin() -> str:
    """Resolve the CORS origin to return for current request."""
    if "*" in ALLOWED_ORIGINS:
        return "*"

    request_origin = request.headers.get("Origin", "")
    if request_origin and request_origin in ALLOWED_ORIGINS:
        return request_origin

    return ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "*"


def _corsify(response):
    """Attach CORS headers for GitHub Pages/local frontend support."""
    origin = _resolve_cors_origin()
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def create_app() -> Flask:
    """Flask app factory for Render/local HTTP API."""
    app = Flask(__name__)

    @app.after_request
    def _after_request(response):
        return _corsify(response)

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({"service": "flashoot-chatbot", "status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})

    @app.route("/chat", methods=["POST", "OPTIONS"])
    def chat_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400

        try:
            logger.info("API request question: %s", message)
            chatbot = _get_chatbot()
            answer = chatbot.ask(message)
            logger.info("API response answer: %s", answer)
            return jsonify({"reply": answer})
        except Exception as e:
            logger.exception("API chat processing failed")
            return jsonify({"error": f"Failed to process request: {e}"}), 500

    return app


if __name__ == "__main__":
    run_mode = os.getenv("RUN_MODE", "cli").strip().lower()
    if run_mode == "api":
        app = create_app()
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    else:
        main()

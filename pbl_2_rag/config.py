"""
HybridBail: System Configuration
Configuration for bail decision system with Gemini/Perplexity support
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class HybridBailConfig:
    """Enhanced configuration class for HybridBail system."""
    
    # ========================================================================
    # DATA SOURCE CONFIGURATION
    # ========================================================================
    PDF_DIRECTORY: str = r"C:\\Me\\Main Bail Cases Folder"
    USE_GOOGLE_DRIVE: bool = False
    GOOGLE_DRIVE_FOLDER_URL: Optional[str] = None
    GOOGLE_DRIVE_FOLDER_ID: Optional[str] = None
    CREDENTIALS_PATH: str = "credentials.json"
    TOKEN_PATH: str = "token.pickle"
    
    # Category folders (17 bail types)
    CATEGORY_FOLDERS: Dict[str, str] = field(default_factory=lambda: {
        "anticipatory_bail": "anticipatory bail",
        "bail_denied": "bail denied",
        "bail_granted": "bail granted",
        "bail_order": "bail order",
        "bail_precedent": "bail precedent",
        "bail_rejected": "bail rejected",
        "bail_with_conditions": "bail with conditions",
        "case_law_bill": "case law bill",
        "crpc_section": "CrPC Section",
        "high_court": "high court",
        "interim_bail": "interim bail",
        "ipc_section": "IPC Section",
        "judicial_discretion_bail": "judicial discretion bail",
        "landmark_bail_judgement": "landmark bail judgement",
        "ndps_bail": "NDPS bail",
        "pocso_bail": "POCSO bail",
        "sc_st_act_bail": "SC,ST Act bail",
        "supreme_court_bail_direction": "supreme court bail direction",
        "uapa_bail": "UAPA bail"
    })
    
    # ========================================================================
    # QDRANT VECTOR DATABASE
    # ========================================================================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    USE_MULTI_COLLECTION: bool = True
    COLLECTION_PREFIX: str = "hybridbail"
    MASTER_COLLECTION_NAME: str = "hybridbail_master"
    VECTOR_SIZE: int = 1024
    DISTANCE_METRIC: str = "Cosine"
    
    # ========================================================================
    # JINA AI EMBEDDINGS
    # ========================================================================
    JINA_API_KEY: Optional[str] = None
    JINA_MODEL: str = "jina-embeddings-v3"
    JINA_DIMENSIONS: int = 1024
    JINA_TASK: str = "retrieval.passage"
    JINA_LATE_CHUNKING: bool = True
    
    # ========================================================================
    # LLM CONFIGURATION (Gemini/Perplexity)
    # ========================================================================
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    LLM_API_KEY: Optional[str] = None
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1
    EVAL_LLM_MODEL: str = "gemini-2.0-flash-exp"
    EVAL_LLM_API_KEY: Optional[str] = None
    
    # ========================================================================
    # PROCESSING PARAMETERS
    # ========================================================================
    CHUNK_SIZE: int = 1000
    OVERLAP_SIZE: int = 200
    MIN_CHUNK_SIZE: int = 100
    MAX_CHUNK_SIZE: int = 2000
    BATCH_SIZE: int = 20
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ========================================================================
    # BAIL DECISION PARAMETERS
    # ========================================================================
    ENABLE_MULTI_LABEL: bool = True
    CLASSIFICATION_THRESHOLD: float = 0.65
    EXTRACT_ATTRIBUTES: bool = True
    AUTO_GRANT_THRESHOLD: float = 0.85
    AUTO_DENY_THRESHOLD: float = 0.15
    HUMAN_INTERVENTION_RANGE: tuple = (0.15, 0.85)
    TOP_K_PRECEDENTS: int = 10
    RERANK_TOP_K: int = 5
    HYBRID_SEARCH_ALPHA: float = 0.6
    
    # ========================================================================
    # MULTILINGUAL
    # ========================================================================
    ENABLE_MULTILINGUAL: bool = True
    DEFAULT_LANGUAGE: str = "en"
    AUTO_DETECT_LANGUAGE: bool = True
    CROSS_LINGUAL_SEARCH: bool = True
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    ENABLE_EVALUATION: bool = True
    EVALUATION_METRICS: List[str] = field(default_factory=lambda: [
        "precision@5", "recall@10", "ndcg@10", 
        "faithfulness", "answer_relevancy", "accuracy", "f1_score"
    ])
    
    # ========================================================================
    # SYSTEM SETTINGS
    # ========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/hybridbail.log"
    ENABLE_TELEMETRY: bool = True
    METADATA_DIR: str = "./data/processed"
    SAVE_INTERMEDIATE_RESULTS: bool = True
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 30
    CACHE_EMBEDDINGS: bool = True
    CACHE_DIR: str = "./cache"
    FORCE_REPROCESS: bool = False
    VERBOSE: bool = True
    
    ATTRIBUTE_WEIGHTS: Dict = field(default_factory=lambda: {
        "crime_category": 0.30,
        "legal_sections": 0.25,
        "age_group": 0.10,
        "gender": 0.05,
        "region": 0.10,
        "criminal_history": 0.15,
        "custody_duration": 0.05
    })
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not installed")
        
        # Determine LLM provider
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        # Set default model
        if llm_provider == "gemini":
            default_model = "gemini-2.0-flash-exp"
        elif llm_provider == "perplexity":
            default_model = "llama-3.1-sonar-large-128k-online"
        else:
            default_model = "gemini-2.0-flash-exp"
        
        # Handle backwards compatibility with old env vars
        #pdf_dir = os.getenv("PDF_DIRECTORY") or os.getenv("DRIVE_FOLDER_URL") or cls.PDF_DIRECTORY
        
        return cls(
            # Data sources
            PDF_DIRECTORY=r"C:\\Me\\Main Bail Cases Folder",
            USE_GOOGLE_DRIVE=os.getenv("USE_GOOGLE_DRIVE", "false").lower() == "true",
            GOOGLE_DRIVE_FOLDER_URL=os.getenv("GOOGLE_DRIVE_FOLDER_URL"),
            GOOGLE_DRIVE_FOLDER_ID=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
            
            # Qdrant
            QDRANT_HOST=os.getenv("QDRANT_HOST", cls.QDRANT_HOST),
            QDRANT_PORT=int(os.getenv("QDRANT_PORT", str(cls.QDRANT_PORT))),
            QDRANT_URL=os.getenv("QDRANT_URL"),
            QDRANT_API_KEY=os.getenv("QDRANT_API_KEY"),
            USE_MULTI_COLLECTION=os.getenv("USE_MULTI_COLLECTION", "true").lower() == "true",
            COLLECTION_PREFIX=os.getenv("COLLECTION_PREFIX", cls.COLLECTION_PREFIX),
            
            # Jina AI
            JINA_API_KEY=os.getenv("JINA_API_KEY"),
            JINA_MODEL=os.getenv("JINA_MODEL", cls.JINA_MODEL),
            JINA_DIMENSIONS=int(os.getenv("JINA_DIMENSIONS", str(cls.JINA_DIMENSIONS))),
            JINA_TASK=os.getenv("JINA_TASK", cls.JINA_TASK),
            JINA_LATE_CHUNKING=os.getenv("JINA_LATE_CHUNKING", "true").lower() == "true",
            
            # LLM
            LLM_PROVIDER=llm_provider,
            LLM_MODEL=os.getenv("LLM_MODEL", default_model),
            LLM_API_KEY=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
            LLM_MAX_TOKENS=int(os.getenv("LLM_MAX_TOKENS", str(cls.LLM_MAX_TOKENS))),
            LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", str(cls.LLM_TEMPERATURE))),
            EVAL_LLM_API_KEY=os.getenv("EVAL_LLM_API_KEY"),
            
            # Processing
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", str(cls.CHUNK_SIZE))),
            OVERLAP_SIZE=int(os.getenv("OVERLAP_SIZE", str(cls.OVERLAP_SIZE))),
            MIN_CHUNK_SIZE=int(os.getenv("MIN_CHUNK_SIZE", str(cls.MIN_CHUNK_SIZE))),
            MAX_CHUNK_SIZE=int(os.getenv("MAX_CHUNK_SIZE", str(cls.MAX_CHUNK_SIZE))),
            BATCH_SIZE=int(os.getenv("BATCH_SIZE", str(cls.BATCH_SIZE))),
            EMBEDDING_BATCH_SIZE=int(os.getenv("EMBEDDING_BATCH_SIZE", str(cls.EMBEDDING_BATCH_SIZE))),
            
            # Thresholds
            AUTO_GRANT_THRESHOLD=float(os.getenv("AUTO_GRANT_THRESHOLD", str(cls.AUTO_GRANT_THRESHOLD))),
            AUTO_DENY_THRESHOLD=float(os.getenv("AUTO_DENY_THRESHOLD", str(cls.AUTO_DENY_THRESHOLD))),
            CLASSIFICATION_THRESHOLD=float(os.getenv("CLASSIFICATION_THRESHOLD", str(cls.CLASSIFICATION_THRESHOLD))),
            
            # Retrieval
            TOP_K_PRECEDENTS=int(os.getenv("TOP_K_PRECEDENTS", str(cls.TOP_K_PRECEDENTS))),
            RERANK_TOP_K=int(os.getenv("RERANK_TOP_K", str(cls.RERANK_TOP_K))),
            HYBRID_SEARCH_ALPHA=float(os.getenv("HYBRID_SEARCH_ALPHA", str(cls.HYBRID_SEARCH_ALPHA))),
            
            # System
            LOG_LEVEL=os.getenv("LOG_LEVEL", cls.LOG_LEVEL),
            LOG_FILE=os.getenv("LOG_FILE", cls.LOG_FILE),
            VERBOSE=os.getenv("VERBOSE", "true").lower() == "true",
            CACHE_EMBEDDINGS=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
            CACHE_DIR=os.getenv("CACHE_DIR", cls.CACHE_DIR),
            FORCE_REPROCESS=os.getenv("FORCE_REPROCESS", "false").lower() == "true",
            METADATA_DIR=os.getenv("METADATA_DIR", cls.METADATA_DIR),
            SAVE_INTERMEDIATE_RESULTS=os.getenv("SAVE_INTERMEDIATE_RESULTS", "true").lower() == "true",
            ENABLE_MULTILINGUAL=os.getenv("ENABLE_MULTILINGUAL", "true").lower() == "true",
            DEFAULT_LANGUAGE=os.getenv("DEFAULT_LANGUAGE", cls.DEFAULT_LANGUAGE),
            ENABLE_EVALUATION=os.getenv("ENABLE_EVALUATION", "true").lower() == "true",
        )
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Required API keys
        if not self.JINA_API_KEY:
            errors.append("JINA_API_KEY is required (get free key from https://jina.ai)")
        
        if not self.LLM_API_KEY:
            if self.LLM_PROVIDER == "gemini":
                errors.append("LLM_API_KEY (Gemini) required (get from https://aistudio.google.com/apikey)")
            else:
                errors.append(f"LLM_API_KEY required for {self.LLM_PROVIDER}")
        
        # Directory checks
        if not os.path.exists(self.PDF_DIRECTORY):
            os.makedirs(self.PDF_DIRECTORY, exist_ok=True)
            logger.info(f"Created PDF directory: {self.PDF_DIRECTORY}")
        
        # Threshold validation
        if not (0 < self.AUTO_GRANT_THRESHOLD <= 1):
            errors.append("AUTO_GRANT_THRESHOLD must be between 0 and 1")
        
        if not (0 <= self.AUTO_DENY_THRESHOLD < 1):
            errors.append("AUTO_DENY_THRESHOLD must be between 0 and 1")
        
        if self.AUTO_DENY_THRESHOLD >= self.AUTO_GRANT_THRESHOLD:
            errors.append("AUTO_DENY_THRESHOLD must be less than AUTO_GRANT_THRESHOLD")
        
        # Processing parameters
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if self.OVERLAP_SIZE >= self.CHUNK_SIZE:
            errors.append("OVERLAP_SIZE must be less than CHUNK_SIZE")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def get_collection_name(self, category: str) -> str:
        """Get Qdrant collection name for a category."""
        if self.USE_MULTI_COLLECTION:
            return f"{self.COLLECTION_PREFIX}_{category}"
        return self.MASTER_COLLECTION_NAME
    
    def get_category_path(self, category: str) -> str:
        """Get filesystem path for a category's PDFs."""
        folder_name = self.CATEGORY_FOLDERS.get(category, category)
        return os.path.join(self.PDF_DIRECTORY, folder_name)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"""HybridBailConfig(
  Collections: {'Multi-category' if self.USE_MULTI_COLLECTION else 'Single'}
  Embeddings: {self.JINA_MODEL} ({self.JINA_DIMENSIONS}D)
  LLM: {self.LLM_PROVIDER.upper()} - {self.LLM_MODEL}
  Multilingual: {self.ENABLE_MULTILINGUAL}
  Auto-grant threshold: {self.AUTO_GRANT_THRESHOLD}
  Auto-deny threshold: {self.AUTO_DENY_THRESHOLD}
)"""


# Backwards compatibility alias
RAGConfig = HybridBailConfig

import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration class for RAG preprocessing pipeline."""
    
    # Local PDF Configuration
    PDF_DIRECTORY: str = "./pdf_dataset"
    USE_GOOGLE_DRIVE: bool = False
    
    # Google Drive Configuration (optional)
    GOOGLE_DRIVE_FOLDER_URL: str = None
    GOOGLE_DRIVE_FOLDER_ID: str = None
    CREDENTIALS_PATH: str = "credentials.json"
    TOKEN_PATH: str = "token.pickle"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = None
    COLLECTION_NAME: str = "legal_documents"
    
    # Jina AI Configuration
    JINA_API_KEY: str = None
    JINA_MODEL: str = "jina-embeddings-v3"
    JINA_DIMENSIONS: int = 1024
    JINA_TASK: str = "retrieval.passage"
    
    # Processing parameters
    CHUNK_SIZE: int = 1000
    OVERLAP_SIZE: int = 200
    BATCH_SIZE: int = 20
    MIN_CHUNK_SIZE: int = 50
    FORCE_REDOWNLOAD: bool = False
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        # Check if using Google Drive or local folder
        drive_url = os.getenv("GOOGLE_DRIVE_FOLDER_URL")
        local_folder = os.getenv("DRIVE_FOLDER_URL") or os.getenv("PDF_DIRECTORY")
        
        # Determine if using Google Drive
        use_google_drive = drive_url is not None and drive_url.startswith("https://")
        
        # Set PDF directory based on source
        pdf_directory = local_folder if not use_google_drive else "./pdf_dataset"
        
        return cls(
            PDF_DIRECTORY=pdf_directory,
            USE_GOOGLE_DRIVE=use_google_drive,
            GOOGLE_DRIVE_FOLDER_URL=drive_url,
            GOOGLE_DRIVE_FOLDER_ID=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
            CREDENTIALS_PATH=os.getenv("CREDENTIALS_PATH", cls.CREDENTIALS_PATH),
            TOKEN_PATH=os.getenv("TOKEN_PATH", cls.TOKEN_PATH),
            QDRANT_HOST=os.getenv("QDRANT_HOST", cls.QDRANT_HOST),
            QDRANT_PORT=int(os.getenv("QDRANT_PORT", str(cls.QDRANT_PORT))),
            QDRANT_URL=os.getenv("QDRANT_URL"),
            COLLECTION_NAME=os.getenv("COLLECTION_NAME", cls.COLLECTION_NAME),
            JINA_API_KEY=os.getenv("JINA_API_KEY"),
            JINA_MODEL=os.getenv("JINA_MODEL", cls.JINA_MODEL),
            JINA_DIMENSIONS=int(os.getenv("JINA_DIMENSIONS", str(cls.JINA_DIMENSIONS))),
            JINA_TASK=os.getenv("JINA_TASK", cls.JINA_TASK),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", str(cls.CHUNK_SIZE))),
            OVERLAP_SIZE=int(os.getenv("OVERLAP_SIZE", str(cls.OVERLAP_SIZE))),
            BATCH_SIZE=int(os.getenv("BATCH_SIZE", str(cls.BATCH_SIZE))),
            MIN_CHUNK_SIZE=int(os.getenv("MIN_CHUNK_SIZE", str(cls.MIN_CHUNK_SIZE))),
            FORCE_REDOWNLOAD=os.getenv("FORCE_REDOWNLOAD", "false").lower() == "true"
        )
    
    def validate(self):
        """Validate configuration parameters."""
        if not self.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is required")
        
        if not os.path.exists(self.PDF_DIRECTORY):
            raise FileNotFoundError(f"PDF directory not found: {self.PDF_DIRECTORY}")
        
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")

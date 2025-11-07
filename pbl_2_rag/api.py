# api.py

"""
HybridBail FastAPI Backend
District Court Bail Decision Support System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from pathlib import Path
import shutil
import uuid
from datetime import datetime

from config import HybridBailConfig
from utils import setup_logging
from pipeline import HybridBailPipeline
from constants import BAIL_CATEGORIES

logger = logging.getLogger(__name__)

# ==================== INITIALIZATION ====================

app = FastAPI(
    title="HybridBail API",
    description="District Court Bail Decision Support System",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize config and pipeline
config = HybridBailConfig.from_env()
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
pipeline = HybridBailPipeline(config)

# Create upload directory
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory processing status tracking
processing_tasks = {}

# ==================== RESPONSE MODELS ====================

class BailDecisionResponse(BaseModel):
    case_id: str
    filename: str
    category: str
    recommendation: str
    confidence: float
    legal_provisions: Dict
    accused_profile: Dict
    similar_precedents: List[Dict]
    precedent_summary: str
    detailed_reasoning: str
    timestamp: str
    processing_metadata: Dict


class ProcessingStatus(BaseModel):
    case_id: str
    status: str
    message: str


class CategoryInfo(BaseModel):
    category_id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


class DocumentIndexResponse(BaseModel):
    status: str
    message: str
    timestamp: str


# ==================== HEALTH & INFO ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "HybridBail API v2.0",
        "docs": "/docs",
        "description": "District Court Bail Decision Support System"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    logger.info("🏥 Health check requested")
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "pdf_ingester": "ready",
            "text_chunker": "ready",
            "embedder": "ready",
            "vector_db": "ready",
            "classifier": "ready",
            "decision_engine": "ready",
            "llm": "ready"
        }
    }


@app.get("/categories", response_model=List[CategoryInfo], tags=["Reference"])
async def get_categories():
    """Get all bail categories."""
    logger.info("📋 Categories requested")
    categories = []
    for cat_id, cat_info in BAIL_CATEGORIES.items():
        categories.append({
            "category_id": cat_id,
            "name": cat_info['name'],
            "description": cat_info.get('description', '')
        })
    return categories


# ==================== DOCUMENT PROCESSING ENDPOINTS ====================

@app.post("/process-bail", response_model=BailDecisionResponse, tags=["Bail Processing"])
async def process_bail_application(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process a bail application PDF and generate decision.
    
    Complete workflow:
    1. Extract PDF text
    2. Parse legal provisions
    3. Extract accused attributes
    4. Chunk document (400 words + 50 word overlap)
    5. Generate 768D embeddings
    6. Search similar precedents
    7. Generate bail decision
    8. Create detailed reasoning
    
    Args:
        file: PDF file of the bail application
        
    Returns:
        Comprehensive bail decision report
    """
    
    # ===== VALIDATION =====
    if not file.filename:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")
    
    filename_lower = file.filename.lower()
    if not filename_lower.endswith('.pdf'):
        logger.error(f"Invalid file type: {filename_lower}")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # ===== GENERATE CASE ID =====
    case_id = f"BAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logger.info(f"📥 New request - Case ID: {case_id}")
    logger.info(f"📄 Filename: {file.filename}")
    
    # ===== SAVE TEMPORARILY =====
    temp_path = UPLOAD_DIR / f"{case_id}.pdf"
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"💾 File saved temporarily: {temp_path}")
        
        # ===== PROCESS WITH PIPELINE =====
        logger.info(f"🔄 Starting pipeline processing...")
        result = pipeline.process_bail_application(str(temp_path))
        
        # ===== FORMAT RESPONSE =====
        response = {
            "case_id": case_id,
            "filename": file.filename,
            "category": result['case_summary']['category'],
            "recommendation": result['decision']['recommendation'],
            "confidence": result['decision']['confidence'],
            "legal_provisions": result['legal_analysis'],
            "accused_profile": result['accused_profile'],
            "similar_precedents": [
                {
                    "case_title": p.get('case_title', 'N/A'),
                    "decision": p.get('decision', 'N/A'),
                    "similarity_score": p.get('score', 0.0)
                }
                for p in result['similar_precedents'][:5]
            ],
            "precedent_summary": result['precedent_analysis']['summary'],
            "detailed_reasoning": result['detailed_reasoning'],
            "timestamp": datetime.now().isoformat(),
            "processing_metadata": result['processing_metadata']
        }
        
        logger.info(f"✅ Processing complete for case {case_id}")
        
        # ===== SCHEDULE CLEANUP =====
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Error processing {case_id}: {e}", exc_info=True)
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post(
    "/process-bail-async",
    response_model=ProcessingStatus,
    tags=["Bail Processing"]
)
async def process_bail_application_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process bail application asynchronously (for large files).
    
    Returns a case_id to check status later.
    """
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    case_id = f"BAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    temp_path = UPLOAD_DIR / f"{case_id}.pdf"
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize status
        processing_tasks[case_id] = {
            "status": "processing",
            "filename": file.filename,
            "started_at": datetime.now().isoformat()
        }
        
        logger.info(f"📥 Async processing started: {case_id}")
        
        # Add background task
        background_tasks.add_task(
            process_bail_background,
            case_id,
            str(temp_path),
            file.filename
        )
        
        return {
            "case_id": case_id,
            "status": "processing",
            "message": "Bail application submitted for processing. Use /status/{case_id} to check progress."
        }
    
    except Exception as e:
        logger.error(f"Error submitting {case_id}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@app.get("/status/{case_id}", tags=["Bail Processing"])
async def get_processing_status(case_id: str):
    """Get processing status for a case."""
    if case_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Case ID not found")
    
    return processing_tasks[case_id]


# ==================== DOCUMENT INDEXING ENDPOINTS ====================

@app.post("/index-documents", response_model=DocumentIndexResponse, tags=["Administration"])
async def index_documents(
    category: Optional[str] = Query(None, description="Specific category to index"),
    background_tasks: BackgroundTasks = None
):
    """
    Index bail case documents into the vector database.
    
    Workflow:
    1. Extract PDFs from category folders
    2. Parse legal provisions
    3. Extract attributes
    4. Chunk documents (400 words + overlap)
    5. Generate 768D embeddings
    6. Store in Qdrant
    
    Args:
        category: Optional specific category (indexes all if not provided)
    """
    
    try:
        if category and category not in BAIL_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        logger.info(f"📚 Document indexing started - Category: {category or 'ALL'}")
        
        # Run indexing in background
        background_tasks.add_task(index_documents_background, category)
        
        return {
            "status": "started",
            "message": f"Document indexing started for: {category or 'all categories'}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error starting document indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.delete("/case/{case_id}", tags=["Administration"])
async def delete_case(case_id: str):
    """Delete a processed case from memory."""
    if case_id in processing_tasks:
        del processing_tasks[case_id]
        logger.info(f"🗑️  Case deleted: {case_id}")
        return {"status": "deleted", "case_id": case_id}
    raise HTTPException(status_code=404, detail="Case ID not found")


# ==================== BACKGROUND TASKS ====================

def process_bail_background(case_id: str, pdf_path: str, filename: str):
    """Background task to process bail application."""
    try:
        logger.info(f"🔄 Starting background processing for {case_id}")
        
        result = pipeline.process_bail_application(pdf_path)
        
        processing_tasks[case_id] = {
            "status": "completed",
            "filename": filename,
            "result": {
                "category": result['case_summary']['category'],
                "recommendation": result['decision']['recommendation'],
                "confidence": result['decision']['confidence'],
                "legal_provisions": result['legal_analysis'],
                "accused_profile": result['accused_profile'],
                "precedent_summary": result['precedent_analysis']['summary'],
                "detailed_reasoning": result['detailed_reasoning']
            },
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Background processing completed for {case_id}")
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for {case_id}: {e}", exc_info=True)
        processing_tasks[case_id] = {
            "status": "failed",
            "filename": filename,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        cleanup_temp_file(Path(pdf_path))


def index_documents_background(category: Optional[str] = None):
    """Background task to index documents."""
    try:
        logger.info(f"📚 Starting document indexing - Category: {category or 'ALL'}")
        pipeline.process_documents(category=category)
        logger.info("✅ Document indexing completed")
    except Exception as e:
        logger.error(f"❌ Document indexing failed: {e}", exc_info=True)


def cleanup_temp_file(file_path: Path):
    """Delete temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️  Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {e}")


# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting HybridBail API Server")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

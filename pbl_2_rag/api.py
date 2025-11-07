from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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

# Initialize FastAPI app
app = FastAPI(
    title="HybridBail API",
    description="District Court Bail Decision Support System",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration and pipeline
config = HybridBailConfig.from_env()
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
pipeline = HybridBailPipeline(config)

# Create upload directory
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Response Models
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


# In-memory storage for processing status
processing_tasks = {}


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "HybridBail API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "pdf_ingester": "ready",
            "vector_db": "ready",
            "llm": "ready",
            "classifier": "ready"
        }
    }


@app.get("/categories", response_model=List[CategoryInfo], tags=["Reference"])
async def get_categories():
    """Get all bail categories."""
    categories = []
    for cat_id, cat_info in BAIL_CATEGORIES.items():
        categories.append({
            "category_id": cat_id,
            "name": cat_info['name'],
            "description": cat_info.get('description', '')
        })
    return categories


@app.post("/process-bail", response_model=BailDecisionResponse, tags=["Bail Processing"])
async def process_bail_application(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process a bail application PDF and generate decision.
    
    - **file**: PDF file of the bail application
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Generate unique case ID
    case_id = f"BAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Save uploaded file temporarily
    temp_path = UPLOAD_DIR / f"{case_id}.pdf"
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing bail application: {file.filename} (Case ID: {case_id})")
        
        # Process the bail application
        result = pipeline.process_bail_application(str(temp_path))
        
        # Format response
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
            "timestamp": datetime.now().isoformat()
        }
        
        # Schedule cleanup of temp file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing bail application: {e}")
        # Cleanup on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/process-bail-async", response_model=ProcessingStatus, tags=["Bail Processing"])
async def process_bail_application_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process a bail application asynchronously (for large files or slow processing).
    
    - **file**: PDF file of the bail application
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
        
        # Add background task
        background_tasks.add_task(process_bail_background, case_id, str(temp_path), file.filename)
        
        return {
            "case_id": case_id,
            "status": "processing",
            "message": "Bail application submitted for processing. Use /status/{case_id} to check progress."
        }
    
    except Exception as e:
        logger.error(f"Error submitting bail application: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@app.get("/status/{case_id}", tags=["Bail Processing"])
async def get_processing_status(case_id: str):
    """Get processing status for a case."""
    if case_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Case ID not found")
    
    return processing_tasks[case_id]


@app.post("/index-documents", tags=["Administration"])
async def index_documents(
    category: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Index bail case documents into the vector database.
    
    - **category**: Optional specific category to index (indexes all if not provided)
    """
    try:
        if category and category not in BAIL_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Run indexing in background
        background_tasks.add_task(index_documents_background, category)
        
        return {
            "status": "started",
            "message": f"Document indexing started for category: {category or 'all'}"
        }
    
    except Exception as e:
        logger.error(f"Error starting document indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.delete("/case/{case_id}", tags=["Administration"])
async def delete_case(case_id: str):
    """Delete a processed case from memory."""
    if case_id in processing_tasks:
        del processing_tasks[case_id]
        return {"status": "deleted", "case_id": case_id}
    raise HTTPException(status_code=404, detail="Case ID not found")


# Background task functions
def process_bail_background(case_id: str, pdf_path: str, filename: str):
    """Background task to process bail application."""
    try:
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
        
    except Exception as e:
        logger.error(f"Background processing failed for {case_id}: {e}")
        processing_tasks[case_id] = {
            "status": "failed",
            "filename": filename,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        # Cleanup temp file
        cleanup_temp_file(Path(pdf_path))


def index_documents_background(category: Optional[str] = None):
    """Background task to index documents."""
    try:
        logger.info(f"Starting document indexing for category: {category or 'all'}")
        pipeline.process_documents(category=category)
        logger.info("Document indexing completed")
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")


def cleanup_temp_file(file_path: Path):
    """Delete temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

"""
AI MedScan - Medical Image Intelligence Platform
FastAPI Backend Application

This is the main entry point for the AI MedScan API server.
Provides endpoints for medical image analysis, Grad-CAM visualization,
and clinical report generation.
"""

import os
import sys
import uuid
import time
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("medscan")

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.preprocessing import MedicalImagePreprocessor
from app.models.classifier import MedicalImageClassifier
from app.services.gradcam import GradCAMGenerator
from app.services.report_generator import ClinicalReportGenerator

# ─────────────────────────────────────────────
# Application Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI MedScan API",
    description="Medical Image Intelligence Platform - AI-Powered Chest X-Ray Analysis",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Initialize ML Pipeline Components
# ─────────────────────────────────────────────

preprocessor = MedicalImagePreprocessor(use_medical_normalization=True)
classifier = MedicalImageClassifier(mode="demo")
gradcam = GradCAMGenerator()
report_generator = ClinicalReportGenerator()

# Store analysis results in memory (use Redis/DB in production)
analysis_store = {}

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    # Mount CSS and JS directories at their URL paths
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    assets_dir = FRONTEND_DIR / "assets"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    timestamp: str
    processing_time_ms: float
    results: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime: str


# ─────────────────────────────────────────────
# Startup / Shutdown Events
# ─────────────────────────────────────────────

startup_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup."""
    global startup_time
    startup_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("  AI MedScan - Medical Image Intelligence Platform")
    logger.info("  Version 2.0.0 | DenseNet121 Architecture")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading ML model...")
        classifier.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model loading deferred: {e}")
        logger.info("Model will load on first request")
    
    logger.info(f"🌐 Frontend directory: {FRONTEND_DIR}")
    logger.info("🚀 Server ready for requests")


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "AI MedScan API - Visit /api/docs for documentation"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    uptime = str(datetime.now() - startup_time) if startup_time else "N/A"
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model_loaded": classifier._loaded,
        "model_mode": classifier.mode,
        "uptime": uptime,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": "AI MedScan Intelligence Platform",
            "architecture": "DenseNet121-CheXNet",
            "num_pathologies": 14
        }
    }


@app.get("/api/model/info")
async def model_info():
    """Get model architecture information."""
    return classifier.get_model_summary()


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    enhance: bool = Query(True, description="Apply CLAHE enhancement"),
    colormap: str = Query("jet", description="Heatmap colormap (jet, hot, inferno, turbo, plasma)")
):
    """
    Analyze a medical image (chest X-ray).
    
    Pipeline:
    1. Image preprocessing (CLAHE, normalization)
    2. Quality assessment
    3. DenseNet121 classification (14 pathologies)
    4. Grad-CAM heatmap generation
    5. Clinical report generation
    
    Returns comprehensive analysis results including predictions,
    heatmaps, quality metrics, and clinical report.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:12].upper()
    
    logger.info(f"📥 New analysis request: {analysis_id}")
    logger.info(f"   File: {file.filename} | Enhance: {enhance} | Colormap: {colormap}")
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: JPEG, PNG, BMP, TIFF, WebP"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(image_bytes) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")
        
        logger.info(f"   Image size: {len(image_bytes) / 1024:.1f} KB")
        
        # ── Step 1: Preprocessing ──
        logger.info("   [1/5] Preprocessing image...")
        preprocess_result = preprocessor.preprocess(image_bytes, apply_enhancement=enhance)
        
        model_input = preprocess_result["model_input"]
        original_image = preprocess_result["original_image"]
        quality_metrics = preprocess_result["quality_metrics"]
        metadata = preprocess_result["metadata"]
        
        logger.info(f"   Quality Score: {quality_metrics['quality_score']}/100 ({quality_metrics['quality_grade']})")
        
        # ── Step 2: Classification ──
        logger.info("   [2/5] Running DenseNet121 classification...")
        prediction_result = classifier.predict(model_input)
        
        assessment = prediction_result.get("assessment", {})
        logger.info(f"   Assessment: {assessment.get('status', 'N/A')}")
        logger.info(f"   Risk Score: {assessment.get('risk_score', 0)}/100")
        
        # ── Step 3: Grad-CAM ──
        logger.info("   [3/5] Generating Grad-CAM heatmaps...")
        heatmap_result = gradcam.generate(
            model_input, original_image,
            class_index=0,
            colormap=colormap,
            alpha=0.4
        )
        
        # ── Step 4: Multi-class heatmaps ──
        logger.info("   [4/5] Generating multi-class attention maps...")
        multi_heatmaps = gradcam.generate_multi_class_heatmaps(
            model_input, original_image, top_k=3, colormap=colormap
        )
        
        # ── Step 5: Report Generation ──
        logger.info("   [5/5] Generating clinical report...")
        clinical_report = report_generator.generate_report(
            predictions=prediction_result,
            quality_metrics=quality_metrics,
            heatmap_stats=heatmap_result.get("heatmap_stats", {}),
            metadata=metadata
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Compile results
        results = {
            "analysis_id": f"MEDSCAN-{analysis_id}",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round(processing_time, 1),
            
            "image_info": {
                "filename": file.filename,
                "size_kb": round(len(image_bytes) / 1024, 1),
                "original_dimensions": metadata.get("original_size", ""),
                "processed_dimensions": metadata.get("processed_size", "")
            },
            
            "quality_assessment": quality_metrics,
            
            "predictions": {
                "findings": prediction_result.get("findings", []),
                "assessment": prediction_result.get("assessment", {}),
                "model_info": prediction_result.get("model_info", {})
            },
            
            "heatmaps": {
                "primary": {
                    "overlay": heatmap_result.get("heatmap_overlay", ""),
                    "raw": heatmap_result.get("heatmap_raw", ""),
                    "contours": heatmap_result.get("attention_contours", ""),
                    "stats": heatmap_result.get("heatmap_stats", {}),
                    "attention_regions": heatmap_result.get("attention_regions", [])
                }
            },
            
            "clinical_report": clinical_report
        }
        
        # Store results
        analysis_store[analysis_id] = results
        
        logger.info(f"✅ Analysis {analysis_id} completed in {processing_time:.0f}ms")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve a previous analysis result."""
    result = analysis_store.get(analysis_id.replace("MEDSCAN-", ""))
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/api/pathologies")
async def list_pathologies():
    """List all detectable pathologies with descriptions."""
    from app.models.classifier import PATHOLOGY_LABELS, PATHOLOGY_INFO
    
    pathologies = []
    for label in PATHOLOGY_LABELS:
        info = PATHOLOGY_INFO.get(label, {})
        pathologies.append({
            "name": label,
            "description": info.get("description", ""),
            "location": info.get("location", ""),
            "urgency": info.get("urgency", ""),
            "icd10": info.get("icd10", "")
        })
    
    return {"pathologies": pathologies, "total": len(pathologies)}


@app.get("/api/stats")
async def get_stats():
    """Get platform statistics."""
    return {
        "total_analyses": len(analysis_store),
        "model_mode": classifier.mode,
        "model_loaded": classifier._loaded,
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF", "WebP"],
        "max_file_size_mb": 50,
        "pathologies_detected": 14,
        "avg_processing_time_ms": (
            round(
                sum(r.get("processing_time_ms", 0) for r in analysis_store.values()) /
                max(len(analysis_store), 1),
                1
            )
        )
    }


# ─────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting AI MedScan server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

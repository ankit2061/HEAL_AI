from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Optional, List, Dict, Any
import os
import shutil
import uuid
import json
import logging
from datetime import datetime, timedelta
import asyncio

from document_processor import process_file, get_patient_by_insurance_id, format_date_relative, DocumentProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("app")

# MongoDB configuration
MONGO_CONFIG = {
    "host": "mongodb://localhost:27017/",
    "db_name": "HealAI",
}

# Create FastAPI app
app = FastAPI(title="HealAi - Medical Claims Assistant")

# Directory for uploaded files
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# MongoDB connection
mongo_client = None

async def get_mongo_client():
    global mongo_client
    if mongo_client is None:
        mongo_client = AsyncIOMotorClient(MONGO_CONFIG["host"])
        try:
            # Test connection
            await mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise HTTPException(status_code=500, detail="Database connection error")
    return mongo_client

async def get_database():
    client = await get_mongo_client()
    return client[MONGO_CONFIG["db_name"]]

# Serve HTML files directly from root directory
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("docs/index.html")

@app.get("/upload2.html", response_class=HTMLResponse)
async def serve_upload():
    return FileResponse("docs/upload2.html")

@app.get("/form.html", response_class=HTMLResponse)
async def serve_form():
    return FileResponse("docs/form.html")

# API endpoints for document processing
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Upload and process a document"""
    if not file:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No file uploaded"}
        )
    
    # Check file type
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            }
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        
        # Process file asynchronously
        important_info, insurance_id = await process_file(file_path)
        
        # Return success response with extracted data
        return JSONResponse({
            "success": True,
            "insurance_id": insurance_id,
            "important_info": important_info,
            "file_id": file_id
        })
    
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=e.status_code)
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": f"Error processing document: {str(e)}"
        }, status_code=500)

@app.get("/api/patient/{insurance_id}")
async def get_patient(insurance_id: str):
    """Get patient details by insurance ID"""
    try:
        patient = await get_patient_by_insurance_id(insurance_id)
        if not patient:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Patient not found"}
            )
        
        # Format date for display
        if "created_at" in patient and patient["created_at"]:
            # Current date is April 20, 2025
            current_date = datetime.fromisoformat("2025-04-20T07:13:00")
            created_date = datetime.fromisoformat(patient["created_at"].replace("Z", "+00:00"))
            
            # Calculate days difference
            days_diff = (current_date - created_date).days
            
            if days_diff == 0:
                patient["created_at_formatted"] = "Today"
            elif days_diff == 1:
                patient["created_at_formatted"] = "Yesterday"
            elif days_diff < 7:
                patient["created_at_formatted"] = f"{days_diff} days ago"
            elif days_diff < 30:
                weeks = days_diff // 7
                patient["created_at_formatted"] = f"{weeks} week{'s' if weeks > 1 else ''} ago"
            else:
                patient["created_at_formatted"] = created_date.strftime("%B %d, %Y")
            
        return JSONResponse(
            content={"success": True, "patient": patient}
        )
    except Exception as e:
        logger.error(f"Error fetching patient data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"}
        )

@app.post("/api/submit-claim")
async def api_submit_claim(request: Request):
    """Submit a medical claim"""
    try:
        form_data = await request.form()
        form_dict = dict(form_data)
        
        # Get database connection
        db = await get_database()
        
        # Create claim document
        claim_data = {
            "insurance_id": form_dict.get("policyNumber"),
            "patient_name": f"{form_dict.get('patientFirstName', '')} {form_dict.get('patientLastName', '')}".strip(),
            "patient_email": form_dict.get("patientEmail"),
            "patient_dob": form_dict.get("patientDOB"),
            "patient_phone": form_dict.get("patientPhone"),
            "insurance_provider": form_dict.get("insuranceProvider"),
            "group_number": form_dict.get("groupNumber"),
            "service_date": form_dict.get("serviceDate"),
            "provider_name": form_dict.get("providerName"),
            "diagnosis": form_dict.get("diagnosis"),
            "claim_amount": float(form_dict.get("claimAmount", 0)),
            "claim_description": form_dict.get("claimDescription"),
            "status": "submitted",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Insert claim into database
        result = await db.claims.insert_one(claim_data)
        
        # Generate claim reference number
        date_str = datetime.now().strftime("%Y%m%d")
        random_num = str(uuid.uuid4().int)[:4]
        claim_reference = f"CLM-{date_str}-{random_num}"
        
        # Update claim with reference number
        await db.claims.update_one(
            {"_id": result.inserted_id},
            {"$set": {"claim_reference": claim_reference}}
        )
        
        return JSONResponse({
            "success": True,
            "message": "Claim submitted successfully",
            "claim_reference": claim_reference
        })
    
    except Exception as e:
        logger.error(f"Error submitting claim: {str(e)}")
        return JSONResponse({
            "success": False,
            "message": f"Error submitting claim: {str(e)}"
        }, status_code=500)

@app.get("/api/processing-status/{file_id}")
async def get_processing_status(file_id: str):
    """Get the status of a file being processed"""
    from document_processor import processing_statuses
    
    if file_id in processing_statuses:
        status = processing_statuses[file_id]
        return {
            "status": status.status,
            "progress": status.progress,
            "message": status.message,
            "result": status.result,
            "error": status.error
        }
    return {"status": "not_found", "message": "No processing status found for this file"}

@app.get("/form")
async def form_page_redirect(insurance_id: str = ""):
    """Redirect to form.html with insurance_id parameter"""
    return RedirectResponse(url=f"form.html?insurance_id={insurance_id}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection on startup"""
    try:
        await get_mongo_client()
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

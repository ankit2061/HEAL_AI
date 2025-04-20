import cv2
import pytesseract
import numpy as np
import spacy
import pandas as pd
import pdfplumber
from PIL import Image
import re
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
import logging
import asyncio
import ssl
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exception classes for better FastAPI integration
class DocumentProcessingError(Exception):
    """Base class for document processing errors"""
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentExtractionError(DocumentProcessingError):
    """Error during text extraction"""
    def __init__(self, message):
        super().__init__(message, status_code=422)

class DocumentValidationError(DocumentProcessingError):
    """Error during document validation"""
    def __init__(self, message):
        super().__init__(message, status_code=400)

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    nlp = None

# MongoDB connection details
MONGO_CONFIG = {
    "host": "mongodb://localhost:27017/",
    "port": 27017,
    "db_name": "HealAI",
}

# Global connection object for connection pooling
_db_client = None

async def get_db_client():
    """Get async MongoDB client with connection pooling"""
    global _db_client
    if _db_client is None:
        try:
            # Use a more robust connection string with explicit options
            connection_string = MONGO_CONFIG["host"]
            
            _db_client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Test the connection
            await _db_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _db_client

async def check_db_connection():
    """Check if MongoDB connection is healthy"""
    try:
        client = await get_db_client()
        # Simple ping to check connection
        await client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False

async def get_patient_by_insurance_id(insurance_id):
    """Fetch patient details by insurance ID"""
    try:
        client = await get_db_client()
        db = client[MONGO_CONFIG["db_name"]]
        patient_details = db.patient_details
        
        # Query by insurance_id
        patient = await patient_details.find_one({"insurance_id": insurance_id})
        if not patient:
            return None
            
        # Convert ObjectId to string for JSON serialization
        patient["_id"] = str(patient["_id"])
        
        # Format date for display
        if "created_at" in patient and patient["created_at"]:
            patient["created_at"] = patient["created_at"].isoformat()
            
        return patient
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        raise DocumentProcessingError(f"Database error: {str(e)}")

def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        kernel = np.ones((1, 1), np.uint8)
        processed_image = cv2.dilate(binary, kernel, iterations=1)
        processed_image = cv2.erode(processed_image, kernel, iterations=1)

        return processed_image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def extract_text_from_image(image_path):
    """Extract text from image using OCR with improved configuration"""
    try:
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return ""

        pil_image = Image.fromarray(preprocessed_image)
        
        # Enhanced Tesseract configuration
        custom_config = r'--oem 3 --psm 6 -l eng+hin'
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""
    return text.strip()

def extract_aadhaar_number(text):
    """Extract Aadhaar numbers using multiple approaches"""
    # Clean the text first - this can help with OCR errors
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Method 1: Look for common Aadhaar number patterns (12 digits with or without separators)
    aadhaar_patterns = [
        # Pattern with no separators - 12 consecutive digits
        r'(?<!\d)(\d{12})(?!\d)',
        # Pattern with space separators
        r'(\d{4}\s+\d{4}\s+\d{4})',
        # Pattern with dash separators
        r'(\d{4}-\d{4}-\d{4})',
        # Pattern with dot separators
        r'(\d{4}\.\d{4}\.\d{4})'
    ]
    
    for pattern in aadhaar_patterns:
        matches = re.findall(pattern, cleaned_text)
        if matches:
            # Clean up the found number (remove spaces, dashes)
            aadhaar = re.sub(r'[^\d]', '', matches[0])
            if len(aadhaar) == 12:
                return aadhaar
    
    # Method 2: Look for Aadhaar numbers with keywords
    keyword_patterns = [
        # Various ways "Aadhaar" might be written followed by a number
        r'(?:aadhar|aadhaar|adhar|aadha+r|आधार)(?:\s*(?:card|number|no|id|#|:|नंबर|संख्या))?\s*[:\.\-]?\s*((?:\d[\d\s\.\-]*){12})',
        r'(?:uid|unique\s+id)(?:\s*(?:number|no|#))?\s*[:\.\-]?\s*((?:\d[\d\s\.\-]*){12})',
        # Looking for "No:" or "Number:" followed by what could be an Aadhaar
        r'(?:no|number|id)?\s*[:\.\-]\s*((?:\d[\d\s\.\-]*){12})'
    ]
    
    for pattern in keyword_patterns:
        matches = re.findall(pattern, cleaned_text.lower())
        if matches:
            # Clean up the found number
            aadhaar = re.sub(r'[^\d]', '', matches[0])
            if len(aadhaar) == 12:
                return aadhaar
    
    # Method 3: More aggressive - find any 12-digit sequence that could be an Aadhaar number
    digit_sequences = re.findall(r'(?<!\d)(\d[\d\s\.\-]*\d)(?!\d)', cleaned_text)
    for seq in digit_sequences:
        digits_only = re.sub(r'[^\d]', '', seq)
        if len(digits_only) == 12:
            return digits_only
            
    return None

def clean_extracted_field(text, field_type):
    """Clean extracted text based on field type to remove common OCR artifacts"""
    # Convert to string in case we received another type
    text = str(text).strip()
    
    # Remove common label text that might be captured within the value
    unwanted_labels = [
        "Phone Number", "Contact", "Mobile", "Call",
        "Hospital Name", "Doctor", "Clinic", "MD", "Dr\\.",
        "Address", "Location", "Place", "Residence",
        "Insurance ID", "Policy Number", "Insurance",
        "Amount", "Total", "Fee", "Payment",
        "Disease", "Diagnosis", "Condition",
        "Medicines", "Medication", "Drugs", "Prescription"
    ]
    
    # For each unwanted label, try to remove it if it appears at the end
    for label in unwanted_labels:
        # Create pattern to match label at the end of the text (allowing for spaces)
        pattern = rf'\s*{re.escape(label)}$'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove common field separators
    text = re.sub(r'[:;|]$', '', text)
    
    # Clean up newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Additional field-specific cleaning
    if field_type in ["Address"]:
        # Keep only relevant address information
        text = re.sub(r'\s*(?:Phone|Mobile|Contact|Email).*$', '', text, flags=re.IGNORECASE)
    
    elif field_type in ["Hospital Name"]:
        # Remove doctor references
        text = re.sub(r'\s*(?:Doctor|Dr\.|MD|Physician).*$', '', text, flags=re.IGNORECASE)
    
    elif field_type in ["Phone Number"]:
        # Keep only digits and basic formatting characters
        text = re.sub(r'[^\d+\-\s()]', '', text)
    
    return text.strip()

def extract_fields_with_boundaries(text):
    """Extract fields with improved boundary detection to prevent label bleed"""
    extracted_info = []
    found_labels = set()
    
    # Dictionary of field patterns with better boundary detection
    field_patterns = {
        "Name": r'(?:Patient(?:\s*Name)?|Name|Patient)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Father|Gender|Blood|Aadhaar))',
        "Father's Name": r'(?:Father(?:[\'s]*\s*Name)?|Father)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Gender|Blood|Aadhaar))',
        "Gender": r'(?:Gender|Sex)[:;]?\s*(Male|Female|Other|M|F)(?=\n|$)',
        "Blood Group": r'(?:Blood(?:\s*Group)?)[:;]?\s*([ABO][+-]|AB[+-])(?=\n|$)',
        "Address": r'(?:Address|Location|Place|Residence)[:;]?\s*([\w\s,\.\-\/]+?)(?=\n|$|(?:Phone|Mobile|Contact|Email))',
        "Hospital Name": r'(?:Hospital(?:\s*Name)?|Clinic|Medical Center)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Doctor|Dr|MD|Address))',
        "Insurance ID": r'(?:Insurance(?:\s*(?:ID|Number|No))?|Policy(?:\s*Number)?)[:;]?\s*([\w\d\-]+?)(?=\n|$)',
        "Phone Number": r'(?:Phone(?:\s*Number)?|Mobile|Contact|Cell)[:;]?\s*([\d\s\+\-\(\)]+?)(?=\n|$)',
        "Amount": r'(?:Amount|Total|Cost|Fee|Charges)[:;]?\s*([\d\.]+?)(?=\n|$|Rs|\$|₹)',
        "Disease Name": r'(?:Disease(?:\s*Name)?|Diagnosis|Condition|Ailment)[:;]?\s*([\w\s]+?)(?=\n|$|(?:Disease Details|Symptoms|Treatment))',
        "Disease Details": r'(?:Disease(?:\s*Details)?|Details|Diagnosis Details|Clinical Details|Symptoms)[:;]?\s*([\w\s,\.;\(\)\-\/]+?)(?=\n\n|\n(?:Medicines|Medications|Drugs)|$)',
        "Medicines": r'(?:Medicines|Medications|Drugs|Prescriptions|Medicine List)[:;]?\s*([\w\s,\.;\(\)\-\/]+?)(?=\n\n|\n(?:Bed|Ventilation|Amount|Charges)|$)',
        "Bed Type": r'(?:Bed(?:\s*Type)?)[:;]?\s*([\w\s]+?)(?=\n|$)',
        "Ventilation": r'(?:Ventilation|Ventilator|Oxygen)[:;]?\s*(Yes|No|Required|Not Required)(?=\n|$)',
        "Other Charges": r'(?:Other(?:\s*Charges)?|Additional(?:\s*Charges)?|Extra)[:;]?\s*([\d\.]+?)(?=\n|$|Rs|\$|₹)'
    }
    
    # 1. First pass: Extract Aadhaar number with dedicated function
    aadhaar = extract_aadhaar_number(text)
    if aadhaar:
        formatted_aadhaar = f"{aadhaar[:4]}-{aadhaar[4:8]}-{aadhaar[8:]}"
        extracted_info.append({"Text": formatted_aadhaar, "Label": "Aadhar Card"})
        found_labels.add("Aadhar Card")
    
    # 2. Second pass: Extract other fields with improved boundary detection
    for label, pattern in field_patterns.items():
        if label in found_labels:
            continue
            
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            extracted_text = matches.group(1).strip()
            # Clean the extracted text to remove potential label contamination
            cleaned_text = clean_extracted_field(extracted_text, label)
            
            # Only add if we have meaningful content
            if cleaned_text and len(cleaned_text) > 0:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    # 3. Third pass: Look for unlabeled numbers that might be specific fields
    if "Phone Number" not in found_labels:
        # Look for potential phone numbers (10-digit sequences)
        phone_matches = re.search(r'(?<!\d)(\d{10})(?!\d)', text)
        if phone_matches:
            extracted_info.append({"Text": phone_matches.group(1), "Label": "Phone Number"})
            found_labels.add("Phone Number")
    
    # Look for Appendicitis or other common conditions if disease name not found
    if "Disease Name" not in found_labels:
        common_diseases = ["appendicitis", "diabetes", "hypertension", "cancer", "fracture", "pneumonia"]
        for disease in common_diseases:
            if re.search(rf'\b{disease}\b', text, re.IGNORECASE):
                extracted_info.append({"Text": disease.capitalize(), "Label": "Disease Name"})
                found_labels.add("Disease Name")
                break
    
    return extracted_info

def process_text(text, keywords=[]):
    """Main processing function that combines extraction methods"""
    # Get fields using improved boundary detection
    extracted_info = extract_fields_with_boundaries(text)
    
    # For backward compatibility, still use keyword-based extraction for any missing fields
    found_labels = {item["Label"] for item in extracted_info}
    
    for keyword in keywords:
        # Skip keywords for fields we already found
        label = keyword.replace(":", "").strip()
        if any(label in existing for existing in found_labels):
            continue
            
        # Simple keyword-based extraction as fallback
        pattern = re.compile(rf"{re.escape(keyword)}\s*([\w\s\d\.\-]+?)(?=\n|$)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            extracted_text = match.group(1).strip()
            cleaned_text = clean_extracted_field(extracted_text, label)
            
            if cleaned_text and len(cleaned_text) > 0:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    return extracted_info

async def save_to_database(data, insurance_id, file_path):
    """Save extracted data to MongoDB asynchronously"""
    try:
        # Connect to MongoDB using async client
        client = await get_db_client()
        db = client[MONGO_CONFIG["db_name"]]
        
        # Create document collections
        patient_documents = db.patient_documents
        patient_details = db.patient_details
        
        # Insert document information
        doc_data = {
            "insurance_id": insurance_id,
            "file_path": file_path,
            "created_at": datetime.now()
        }
        
        doc_result = await patient_documents.insert_one(doc_data)
        
        # Create patient details document
        patient_data = {
            "insurance_id": insurance_id,
            "name": next((item["Text"] for item in data if item["Label"] == "Name"), None),
            "father_name": next((item["Text"] for item in data if item["Label"] == "Father's Name"), None),
            "aadhar_card": next((item["Text"] for item in data if item["Label"] == "Aadhar Card"), None),
            "gender": next((item["Text"] for item in data if item["Label"] == "Gender"), None),
            "blood_group": next((item["Text"] for item in data if item["Label"] == "Blood Group"), None),
            "address": next((item["Text"] for item in data if item["Label"] == "Address"), None),
            "hospital_name": next((item["Text"] for item in data if item["Label"] == "Hospital Name"), None),
            "phone_number": next((item["Text"] for item in data if item["Label"] == "Phone Number"), None),
            "disease_name": next((item["Text"] for item in data if item["Label"] == "Disease Name"), None),
            "disease_details": next((item["Text"] for item in data if item["Label"] == "Disease Details"), None),
            "medicines": next((item["Text"] for item in data if item["Label"] == "Medicines"), None),
            "bed_type": next((item["Text"] for item in data if item["Label"] == "Bed Type"), None),
            "created_at": datetime.now()
        }
        
        # Handle numeric fields
        amount = next((item["Text"] for item in data if item["Label"] == "Amount"), None)
        if amount:
            amount = re.sub(r'[^\d.]', '', amount)
            if amount:
                try:
                    patient_data["amount"] = float(amount)
                except ValueError:
                    patient_data["amount"] = None
                    
        other_charges = next((item["Text"] for item in data if item["Label"] == "Other Charges"), None)
        if other_charges:
            other_charges = re.sub(r'[^\d.]', '', other_charges)
            if other_charges:
                try:
                    patient_data["other_charges"] = float(other_charges)
                except ValueError:
                    patient_data["other_charges"] = None
        
        # Check if a patient with this insurance_id already exists
        existing_patient = await patient_details.find_one({"insurance_id": insurance_id})
        
        if existing_patient:
            # Update existing patient record
            result = await patient_details.update_one(
                {"insurance_id": insurance_id},
                {"$set": patient_data}
            )
            logger.info(f"Updated existing patient record for insurance ID: {insurance_id}")
            return True
        else:
            # Insert new patient record
            patient_result = await patient_details.insert_one(patient_data)
            logger.info(f"Inserted new patient record with ID: {patient_result.inserted_id}")
            return True
            
    except Exception as err:
        logger.error(f"Error saving to MongoDB: {err}")
        raise DocumentProcessingError(f"Database error: {str(err)}")

async def cleanup_file(file_path):
    """Remove temporary file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up file {file_path}: {e}")

# Progress tracking for long-running document processing
class ProcessingStatus:
    def __init__(self, file_id):
        self.file_id = file_id
        self.status = "pending"
        self.progress = 0
        self.message = "Initializing..."
        self.result = None
        self.error = None

# Global dictionary to store processing status
processing_statuses = {}

async def process_file(file_path, timeout=60):
    """Process uploaded file with timeout"""
    try:
        return await asyncio.wait_for(
            _process_file_internal(file_path),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Processing timed out for file: {file_path}")
        raise DocumentProcessingError("Document processing timed out. Please try again with a simpler document.", 408)

async def _process_file_internal(file_path):
    """Process uploaded file and extract information asynchronously"""
    try:
        text = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image processing is CPU-bound, run in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_text_from_image, file_path)
        elif file_path.lower().endswith('.pdf'):
            # PDF processing is also CPU-bound
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_text_from_pdf, file_path)
        else:
            logger.error(f"Unsupported file type: {file_path}")
            raise DocumentValidationError("Unsupported file type. Only PDF, PNG, JPG, and JPEG are supported.")

        if not text.strip():
            logger.error(f"No text detected in file: {file_path}")
            raise DocumentExtractionError("No text could be extracted from the document.")

        logger.info(f"Extracted text from {file_path}")

        # Extract important information - run in thread pool as it's CPU-bound
        loop = asyncio.get_event_loop()
        important_info = await loop.run_in_executor(
            None, 
            lambda: process_text(text, [
                "Name:", "Father's Name:", "Aadhar Card:", "Gender:", "Blood Group:", 
                "Address:", "Hospital Name:", "Insurance ID:", "Phone Number:", 
                "Amount:", "Disease Name:", "Disease Details:", "Medicines:", 
                "Bed Type:", "Ventilation:", "Other Charges:"
            ])
        )
        
        if not important_info:
            logger.error(f"No important information found in file: {file_path}")
            raise DocumentExtractionError("Could not extract any relevant information from the document.")

        # Get insurance ID
        insurance_id = next((item["Text"] for item in important_info if item["Label"] == "Insurance ID"), None)

        # If no insurance ID found, generate one
        if not insurance_id:
            logger.warning(f"Insurance ID not found in file: {file_path}, generating one")
            # Generate a random insurance ID with INS prefix and 8 digits
            insurance_id = f"INS{uuid.uuid4().hex[:8].upper()}"
            important_info.append({"Text": insurance_id, "Label": "Insurance ID"})

        # Save to database
        await save_to_database(important_info, insurance_id, file_path)
        
        return important_info, insurance_id
        
    except DocumentProcessingError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise DocumentProcessingError(f"Error processing file: {str(e)}")

async def process_file_with_status(file_path):
    """Process file with status tracking"""
    file_id = os.path.basename(file_path)
    processing_statuses[file_id] = ProcessingStatus(file_id)
    
    try:
        # Update status
        processing_statuses[file_id].status = "processing"
        processing_statuses[file_id].message = "Extracting text..."
        processing_statuses[file_id].progress = 10
        
        # Process file
        result = await process_file(file_path)
        
        # Update status
        processing_statuses[file_id].status = "completed"
        processing_statuses[file_id].progress = 100
        processing_statuses[file_id].message = "Processing complete"
        processing_statuses[file_id].result = result
        
        return result
    except Exception as e:
        # Update status
        processing_statuses[file_id].status = "failed"
        processing_statuses[file_id].error = str(e)
        processing_statuses[file_id].message = f"Processing failed: {str(e)}"
        raise

async def format_date_relative(date_str):
    """Format date as relative to current date (e.g., '2 days ago')"""
    try:
        # Parse the ISO date string
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        # Get current date
        now = datetime.now()
        
        # Calculate difference in days
        delta = now - date_obj
        days = delta.days
        
        if days == 0:
            return "Today"
        elif days == 1:
            return "Yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    except Exception as e:
        logger.error(f"Error formatting date: {e}")
        return date_str

# Configure Tesseract path if needed
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif os.name == 'posix':  # macOS/Linux
    if os.path.exists('/opt/homebrew/Caskroom/miniconda/base/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Caskroom/miniconda/base/bin/tesseract'
    elif os.path.exists('/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

import os
import streamlit as st
from utils.document_processor import process_upload
from utils.ai_service import get_gemini_explanation
from utils.patient_data import get_patient_data, validate_field_name, validate_insurance_id
import re

# Set page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "I can help with:\n\n* Medical term explanations\n* Insurance information\n* Patient data queries"
    }]
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {}

# Main Layout
# Sidebar for Patient Management
with st.sidebar:
    st.title("Patient Management")
    
    # Document Upload Section
    st.header("Upload Medical Document")
    uploaded_file = st.file_uploader("Drag and drop file here", 
                                     type=["pdf", "png", "jpg", "jpeg"],
                                     label_visibility="collapsed")
    
    st.caption("Limit 200MB per file • PDF, PNG, JPG, JPEG")
    
    # Process uploaded document
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document..."):
            try:
                fields, text = process_upload(temp_path)
                if fields and fields.get("patient_id") != "Not found":
                    patient_id = fields.get("patient_id")
                    st.session_state.processed_data[patient_id] = {
                        "fields": fields,
                        "text": text,
                        "filename": uploaded_file.name
                    }
                    st.session_state.current_patient = patient_id
                    st.success(f"Processed document for Patient {patient_id}")
                else:
                    st.error("Failed to extract patient ID from document")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Patient Selection (only show if patients exist)
    if st.session_state.processed_data:
        st.header("Select Patient")
        patient_ids = list(st.session_state.processed_data.keys())
        selected_patient = st.selectbox("", patient_ids, 
                                       index=patient_ids.index(st.session_state.current_patient) if st.session_state.current_patient in patient_ids else 0,
                                       label_visibility="collapsed")
        if selected_patient and selected_patient != st.session_state.current_patient:
            st.session_state.current_patient = selected_patient
            st.rerun()

# Main Content Area
st.title("Medical AI Assistant")

# Patient Information Section (at the top when available)
if st.session_state.current_patient:
    patient_data = st.session_state.processed_data.get(st.session_state.current_patient)
    if patient_data:
        st.header("Current Patient Overview")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Name", patient_data["fields"].get("name", "Not found"))
        with info_cols[1]:
            st.metric("Age", patient_data["fields"].get("age", "Not found"))
        with info_cols[2]:
            st.metric("Condition", patient_data["fields"].get("disease", "Not found"))
        
        with st.expander("View Full Document Text"):
            st.text(patient_data["text"])

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at root level
prompt = st.chat_input("Ask about medical terms or patient information")

# Process user input when provided
if prompt:
    # Add user message to UI and state
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the query
    query = prompt.strip().lower()
    response = None  # Initialize response

    if any(keyword in query for keyword in ["insurance", "policy", "claim", "deductible", "premium"]):
        prompt_text = f"Explain the insurance term '{query}'. Provide a clear and concise explanation."
        response = get_gemini_explanation(prompt_text)

    elif any(keyword in query for keyword in ["medical", "disease", "medicine", "hypertension", "diabetes", "appendicitis"]):
        prompt_text = f"Explain the medical term '{query}'. Provide a clear and concise explanation."
        response = get_gemini_explanation(prompt_text)

    elif any(keyword in query for keyword in ["data", "information", "details", "name", "father", "aadhar", "gender", "blood", "address", "hospital", "phone", "disease", "medicines", "bed", "amount", "charges"]):
        insurance_id_match = re.search(r'\b[A-Za-z0-9-]+\b', query)

        if insurance_id_match:
            insurance_id = insurance_id_match.group(0)

            if not validate_insurance_id(insurance_id):
                response = "Invalid Insurance ID format. Please use a valid ID."
            else:
                field_match = re.search(r'(name|father|aadhar|gender|blood|address|hospital|phone|disease|medicines|bed|amount|charges)', query, re.IGNORECASE)

                if field_match:
                    field = field_match.group(0).lower()
                    if not validate_field_name(field):
                        response = f"Invalid field name. Choose from: name, father, aadhar, gender, blood, address, hospital, phone, disease, medicines, bed, amount, charges."
                    else:
                        response = get_patient_data(insurance_id, field)
                else:
                    response = "Please specify a field (name, address, etc.)."
        else:
            response = "Please provide an Insurance ID."

    else:
        response = "I can help explain medical and insurance terms, and retrieve data. Please specify what you are looking for."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

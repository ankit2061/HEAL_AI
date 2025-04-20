import streamlit as st
import google.generativeai as genai
import json
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime
import threading
import schedule
import time
import yaml
import os
from utils import fetch_irdai_data, fetch_claim_settlement_data, scrape_premium_data, fetch_terms_and_conditions

# Configure page
st.set_page_config(
    page_title="Health Insurance Advisor",
    page_icon="🏥",
    layout="wide"
)

# Get API key from environment or use fallback 
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAKzXbzsAo_LBNenAotZX90EHY35nKC2Xc")

# Configure Gemini API with the API key
genai.configure(api_key=API_KEY)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "age": None,
        "gender": None,
        "pre_existing_conditions": [],
        "family_size": None,
        "budget": None,
        "coverage_amount": None,
        "preferred_features": []
    }

if "latest_irdai_data" not in st.session_state:
    st.session_state.latest_irdai_data = []

if "claim_settlement_data" not in st.session_state:
    st.session_state.claim_settlement_data = []

if "last_update" not in st.session_state:
    st.session_state.last_update = None

if "insurance_recommendations" not in st.session_state:
    st.session_state.insurance_recommendations = []


# Load insurance database from YAML file
def load_insurance_database():
    try:
        with open("insurance_database.yml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading insurance database: {str(e)}")
        return []


# Load the insurance database
INSURANCE_DATABASE = load_insurance_database()


# Function to create Gemini model
def get_gemini_model():
    """Create and return a Gemini model instance."""
    try:
        # Using gemini-2.0-flash model for better performance
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None


# Function with exponential backoff for API calls
def generate_with_backoff(model, prompt, max_retries=3):
    """Make API calls with exponential backoff for rate limiting."""
    retries = 0
    while retries < max_retries:
        try:
            return model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e) or "quota" in str(e).lower():
                wait_time = (2 ** retries) * 5  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e

    # If we've exhausted retries, return a graceful message
    class FallbackResponse:
        def __init__(self):
            self.text = "I'm currently experiencing high demand. Please try again in a few minutes."

    return FallbackResponse()


# Function to run data update jobs in the background
def start_background_jobs():
    def run_scheduled_jobs():
        schedule.every(24).hours.do(fetch_irdai_data)  
        schedule.every(24).hours.do(fetch_claim_settlement_data)

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    # Start the background thread
    bg_thread = threading.Thread(target=run_scheduled_jobs)
    bg_thread.daemon = True  # Daemon thread will close when the main program exits
    bg_thread.start()


# Function to get personalized insurance recommendations using Gemini
def get_insurance_recommendations(user_profile, insurance_database):
    try:
        model = get_gemini_model()
        if not model:
            return []

        # Convert user profile to a readable format
        profile_str = "\n".join([f"{k}: {v}" for k, v in user_profile.items() if v])
        
        # Convert insurance database to a readable format
        insurance_str = yaml.dump(insurance_database)
        
        # Create prompt for the AI model
        prompt = f"""
        As a health insurance advisor, recommend the best health insurance policies based on the following user profile:
        
        USER PROFILE:
        {profile_str}
        
        AVAILABLE INSURANCE POLICIES:
        {insurance_str}
        
        Please recommend the top 3 most suitable insurance policies for this user. For each recommendation, provide:
        1. Insurance company name
        2. Policy name
        3. Why this is suitable for the user
        4. Key benefits
        5. Any limitations or considerations
        6. Approximate premium estimate based on the user profile
        
        Format your response as a structured JSON with these fields:
        {{
            "recommendations": [
                {{
                    "rank": 1,
                    "company": "Company name",
                    "policy": "Policy name", 
                    "suitability_reason": "Why this is suitable",
                    "key_benefits": ["benefit1", "benefit2", "benefit3"],
                    "limitations": ["limitation1", "limitation2"],
                    "premium_estimate": "Estimated premium range"
                }},
                // more recommendations...
            ]
        }}
        """
        
        response = generate_with_backoff(model, prompt)
        
        # Parse and return the recommendations
        response_text = response.text
        
        # Extract JSON part if there's any explanatory text around it
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            json_str = json_match.group(0)
            recommendations = json.loads(json_str)
            return recommendations.get("recommendations", [])
        
        return []
    except Exception as e:
        st.error(f"Error getting insurance recommendations: {str(e)}")
        return []


# Function to compare insurance policies
def compare_insurance_policies(policies, insurance_database):
    try:
        model = get_gemini_model()
        if not model:
            return ""

        # Extract policy details from database
        policy_details = []
        for policy_name in policies:
            for company in insurance_database:
                for policy in company.get("policies", []):
                    if policy["name"] == policy_name:
                        policy_details.append({
                            "company": company["name"],
                            "policy": policy["name"],
                            "details": policy
                        })

        if not policy_details:
            return "No policy details found for comparison."

        # Create prompt for the AI model
        policies_str = yaml.dump(policy_details)
        prompt = f"""
        Compare the following health insurance policies objectively:
        
        {policies_str}
        
        Please provide a detailed comparison including:
        1. Premium cost comparison
        2. Coverage benefits comparison
        3. Waiting periods comparison
        4. Special features comparison
        5. Pros and cons of each policy
        6. Which policy might be better for different types of users
        
        Format your response in a clear, structured way with headings and bullet points.
        """
        
        response = generate_with_backoff(model, prompt)
        return response.text
    except Exception as e:
        st.error(f"Error comparing insurance policies: {str(e)}")
        return f"Error comparing policies: {str(e)}"


# Function to answer health insurance related questions
def answer_insurance_question(question):
    try:
        model = get_gemini_model()
        if not model:
            return "Sorry, I'm unable to answer your question at the moment."

        # Create a prompt with context about the question being insurance-related
        prompt = f"""
        As a health insurance expert, please answer the following question about health insurance:
        
        Question: {question}
        
        Provide a detailed, accurate, and helpful answer based on your knowledge of health insurance in India.
        Include relevant facts, regulations, and practical advice where appropriate.
        """
        
        response = generate_with_backoff(model, prompt)
        return response.text
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return f"I'm sorry, I encountered an error while answering your question. Please try again."


# Main application UI
def main():
    # Create sidebar for user profile
    st.sidebar.title("User Profile")
    
    with st.sidebar.form("user_profile_form"):
        st.write("Enter your details for personalized recommendations")
        
        age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.user_profile["age"] if st.session_state.user_profile["age"] else 30)
        
        gender = st.selectbox(
            "Gender", 
            options=["Male", "Female", "Other"],
            index=0 if not st.session_state.user_profile["gender"] else ["Male", "Female", "Other"].index(st.session_state.user_profile["gender"])
        )
        
        pre_existing_conditions = st.multiselect(
            "Pre-existing conditions",
            options=["None", "Diabetes", "Hypertension", "Heart Disease", "Asthma", "Thyroid", "Cancer", "Other"],
            default=st.session_state.user_profile["pre_existing_conditions"] if st.session_state.user_profile["pre_existing_conditions"] else ["None"]
        )
        
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=st.session_state.user_profile["family_size"] if st.session_state.user_profile["family_size"] else 1)
        
        budget = st.slider(
            "Monthly Budget (₹)",
            min_value=1000,
            max_value=50000,
            step=1000,
            value=st.session_state.user_profile["budget"] if st.session_state.user_profile["budget"] else 5000
        )
        
        coverage_amount = st.select_slider(
            "Coverage Amount (₹)",
            options=["₹2 Lakhs", "₹3 Lakhs", "₹5 Lakhs", "₹10 Lakhs", "₹20 Lakhs", "₹50 Lakhs", "₹1 Crore"],
            value=st.session_state.user_profile["coverage_amount"] if st.session_state.user_profile["coverage_amount"] else "₹5 Lakhs"
        )
        
        preferred_features = st.multiselect(
            "Preferred Features",
            options=["Cashless Hospitalization", "No Claim Bonus", "Maternity Benefits", "Critical Illness Cover", "Pre & Post Hospitalization", "Day Care Procedures", "Domiciliary Treatment", "Free Health Check-up"],
            default=st.session_state.user_profile["preferred_features"] if st.session_state.user_profile["preferred_features"] else ["Cashless Hospitalization", "No Claim Bonus"]
        )
        
        submit_button = st.form_submit_button(label="Update Profile & Get Recommendations")
        
        if submit_button:
            # Update session state with new values
            st.session_state.user_profile["age"] = age
            st.session_state.user_profile["gender"] = gender
            st.session_state.user_profile["pre_existing_conditions"] = pre_existing_conditions
            st.session_state.user_profile["family_size"] = family_size
            st.session_state.user_profile["budget"] = budget
            st.session_state.user_profile["coverage_amount"] = coverage_amount
            st.session_state.user_profile["preferred_features"] = preferred_features
            
            # Get recommendations based on updated profile
            with st.spinner("Getting personalized recommendations..."):
                recommendations = get_insurance_recommendations(st.session_state.user_profile, INSURANCE_DATABASE)
                st.session_state.insurance_recommendations = recommendations
    
    # Main content area with tabs
    st.title("Health Insurance Advisor 🏥")
    
    tabs = st.tabs(["Recommendations", "Insurance Policies", "Market Data", "Policy Comparison", "Chat Assistant"])
    
    # Recommendations Tab
    with tabs[0]:
        st.header("Personalized Insurance Recommendations")
        
        if not st.session_state.user_profile["age"]:
            st.info("Please fill out your profile in the sidebar to get personalized recommendations.")
        elif not st.session_state.insurance_recommendations:
            if submit_button:
                st.info("Based on your profile, we're preparing recommendations. Please wait...")
            else:
                st.info("Click 'Update Profile & Get Recommendations' in the sidebar to see your personalized recommendations.")
        else:
            for i, rec in enumerate(st.session_state.insurance_recommendations):
                with st.expander(f"#{rec.get('rank', i+1)}: {rec.get('company', 'Unknown')} - {rec.get('policy', 'Unknown')}", expanded=i==0):
                    st.subheader("Why this is suitable for you")
                    st.write(rec.get("suitability_reason", "No specific reason provided"))
                    
                    st.subheader("Key Benefits")
                    for benefit in rec.get("key_benefits", []):
                        st.write(f"• {benefit}")
                    
                    st.subheader("Limitations")
                    for limitation in rec.get("limitations", []):
                        st.write(f"• {limitation}")
                    
                    st.subheader("Estimated Premium")
                    st.write(rec.get("premium_estimate", "Premium estimate not available"))
                    
                    # Find more details in the database
                    for company in INSURANCE_DATABASE:
                        if company["name"] == rec.get("company"):
                            for policy in company["policies"]:
                                if policy["name"] == rec.get("policy"):
                                    st.subheader("Additional Details")
                                    st.write(f"• Coverage Range: {policy.get('coverage_range', 'Not specified')}")
                                    st.write(f"• Pre-existing Waiting Period: {policy.get('pre_existing_waiting_period', 'Not specified')}")
                                    st.write(f"• Co-payment: {policy.get('co_payment', 'Not specified')}")
                                    st.write(f"• Maternity Coverage: {policy.get('maternity_coverage', 'Not specified')}")
                                    break
    
    # Insurance Policies Tab
    with tabs[1]:
        st.header("All Available Insurance Policies")
        
        # Allow filtering
        col1, col2 = st.columns(2)
        with col1:
            filter_company = st.multiselect(
                "Filter by Insurance Company",
                options=[company["name"] for company in INSURANCE_DATABASE],
                default=[]
            )
        
        with col2:
            coverage_options = []
            for company in INSURANCE_DATABASE:
                for policy in company.get("policies", []):
                    if "coverage_range" in policy:
                        coverage_options.append(policy["coverage_range"])
            
            filter_coverage = st.multiselect(
                "Filter by Coverage Range",
                options=list(set(coverage_options)),
                default=[]
            )
        
        # Display filtered policies
        filtered_companies = INSURANCE_DATABASE
        if filter_company:
            filtered_companies = [company for company in INSURANCE_DATABASE if company["name"] in filter_company]
        
        for company in filtered_companies:
            st.subheader(company["name"])
            st.write(f"Claim Settlement Ratio: {company.get('claim_settlement_ratio', 'Not available')}")
            st.write(f"Cashless Hospitals: {company.get('cashless_hospitals', 'Not available')}")
            
            for policy in company.get("policies", []):
                # Apply coverage filter
                if filter_coverage and policy.get("coverage_range") not in filter_coverage:
                    continue
                    
                with st.expander(policy["name"]):
                    for key, value in policy.items():
                        if key != "name":
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    
    # Market Data Tab
    with tabs[2]:
        st.header("Latest Market Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("IRDAI Recent Updates")
            if st.button("Refresh IRDAI Data"):
                with st.spinner("Fetching latest IRDAI data..."):
                    irdai_data = fetch_irdai_data()
                    st.session_state.latest_irdai_data = irdai_data
                    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("IRDAI data updated!")
            
            if st.session_state.latest_irdai_data:
                st.write(f"Last Updated: {st.session_state.last_update}")
                for item in st.session_state.latest_irdai_data[:10]:  # Show only the first 10
                    st.write(f"**{item.get('company', 'Unknown Company')}**: {item.get('policy', 'Unknown Policy')}")
                    st.write(f"Date: {item.get('date', 'Unknown')}")
                    if item.get('pdf_link'):
                        st.write(f"[View Document]({item.get('pdf_link')})")
                    st.write("---")
            else:
                st.info("Click 'Refresh IRDAI Data' to fetch the latest updates from IRDAI.")
        
        with col2:
            st.subheader("Claim Settlement Ratios")
            if st.button("Refresh Claim Settlement Data"):
                with st.spinner("Fetching latest claim settlement data..."):
                    claim_data = fetch_claim_settlement_data()
                    st.session_state.claim_settlement_data = claim_data
                    st.success("Claim settlement data updated!")
            
            if st.session_state.claim_settlement_data:
                # Create a DataFrame for better display
                claim_df = pd.DataFrame(st.session_state.claim_settlement_data)
                st.dataframe(claim_df)
            else:
                st.info("Click 'Refresh Claim Settlement Data' to fetch the latest claim settlement ratios.")
    
    # Policy Comparison Tab
    with tabs[3]:
        st.header("Compare Insurance Policies")
        
        # Get all policies for selection
        all_policies = []
        for company in INSURANCE_DATABASE:
            for policy in company.get("policies", []):
                all_policies.append(f"{company['name']} - {policy['name']}")
        
        # Allow selecting policies to compare
        selected_policies = st.multiselect(
            "Select policies to compare (2-3 recommended)",
            options=all_policies,
            default=[]
        )
        
        if len(selected_policies) >= 2:
            if st.button("Compare Policies"):
                with st.spinner("Generating comparison..."):
                    # Extract just the policy names
                    policy_names = [policy.split(" - ")[1] for policy in selected_policies]
                    comparison = compare_insurance_policies(policy_names, INSURANCE_DATABASE)
                    st.markdown(comparison)
        else:
            st.info("Please select at least 2 policies to compare.")
    
    # Chat Assistant Tab
    with tabs[4]:
        st.header("Insurance Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        prompt = st.chat_input("Ask me about health insurance...")
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_insurance_question(prompt)
                    st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


# Start background jobs and run main app
if __name__ == "__main__":
    # Start background data update tasks
    start_background_jobs()
    
    # Try to fetch data on startup if not already available
    if not st.session_state.latest_irdai_data:
        fetch_irdai_data()
    
    if not st.session_state.claim_settlement_data:
        fetch_claim_settlement_data()
    
    # Run main application
    main()

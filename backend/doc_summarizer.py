import streamlit as st
import google.generativeai as genai
import PyPDF2
import os
import tempfile

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

class GeminiSummarizer:
    def extract_text_from_pdf(self, file):
        """Extract text content from a PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + " "

            return text
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")

    def summarise_text(self, text, document_type, words_limit=5000, summary_ratio=2/1):
        """Summarize text using Gemini AI with fallback option"""
        try:
            # Limit text to words_limit if needed
            words = text.split()
            if len(words) > words_limit:
                text = ' '.join(words[:words_limit])

            # Calculate the target summary length based on the ratio
            total_lines = text.count('.') + 1
            target_lines = max(1, int(total_lines * summary_ratio))

            model_name = "models/gemini-1.5-flash"
            model = genai.GenerativeModel(model_name)

            # Different prompts based on document type
            if document_type == "medical":
                prompt = f"""Summarise the following medical document in clear, concise language.

Your summary should:
- Be approximately {target_lines} lines long (about {int(summary_ratio * 100)}% of the original text length)
- Maintain accuracy of medical information and terminology while being accessible
- Include the main diagnoses, treatments, recommendations, and conclusions
- Be well-structured with clear sections for different aspects of the document
- Preserve any critical medical values, dosages, or specific instructions

Now summarize this medical text:
{text}"""
            elif document_type == "insurance":
                prompt = f"""Summarise the following insurance document in clear, accessible language for a non-insurance audience.

Your summary should:
- Be approximately {target_lines} lines long (about {int(summary_ratio * 100)}% of the original text length)
- Simplify insurance terminology while maintaining accuracy of the content
- Highlight key terms, conditions, rights, obligations, and important clauses
- Identify any deadlines, requirements, or actions needed
- Structure the summary in clear sections that follow the logical organization of the document
- Emphasize what the average person needs to understand from this document

Now summarize this insurance text:
{text}"""
            else:  # General document
                prompt = f"""Summarise the following document in clear, concise language.

Your summary should:
- Be approximately {target_lines} lines long (about {int(summary_ratio * 100)}% of the original text length)
- Maintain accuracy of the information while making it accessible
- Include all main points, key arguments, and important details
- Be well-structured with clear sections for different aspects of the document
- Preserve any critical values, deadlines, or specific instructions

Now summarize this text:
{text}"""

            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # If the first attempt fails, try with a different model
            try:
                st.warning("Primary model failed, using backup model...")
                backup_model_name = "models/gemini-2.0-flash"
                backup_model = genai.GenerativeModel(backup_model_name)
                response = backup_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e2:
                raise Exception(f"Primary error: {str(e)}\nBackup error: {str(e2)}")

    def chat_with_text(self, text, user_query, document_type):
        """Generate answers to questions about the text using Gemini AI"""
        try:
            model_name = "models/gemini-1.5-flash"
            model = genai.GenerativeModel(model_name)

            # Different prompts based on document type
            if document_type == "medical":
                prompt = f"""You are a medical document assistant. Based on the following medical document, please answer the user's question.

Document content:
{text}

User's question:
{user_query}

Please provide a helpful, accurate response based solely on the information in the document.
If the document doesn't contain information to answer the question, clearly state that you cannot find relevant information in the document.
For medical advice, dosages, or critical information, be precise and quote the exact text from the document when possible.
"""
            elif document_type == "insurance":
                prompt = f"""You are a insurance document assistant. Based on the following insurance document, please answer the user's question.

Document content:
{text}

User's question:
{user_query}

Please provide a helpful, accurate response based solely on the information in the document.
If the document doesn't contain information to answer the question, clearly state that you cannot find relevant information in the document.
For insurance clauses, terms, conditions, or critical provisions, be precise and quote the exact text from the document when possible.
Explain insurance terminology in simple terms while maintaining accuracy.
"""
            else:  # General document
                prompt = f"""You are a document assistant. Based on the following document, please answer the user's question.

Document content:
{text}

User's question:
{user_query}

Please provide a helpful, accurate response based solely on the information in the document.
If the document doesn't contain information to answer the question, clearly state that you cannot find relevant information in the document.
For important facts, figures, or critical information, be precise and quote the exact text from the document when possible.
"""

            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            try:
                st.warning("Primary model failed, using backup model...")
                backup_model_name = "models/gemini-2.0-flash"
                backup_model = genai.GenerativeModel(backup_model_name)
                response = backup_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e2:
                raise Exception(f"Primary error: {str(e)}\nBackup error: {str(e2)}")


def main():
    st.set_page_config(
        page_title="Document Summarizer",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Document Summarizer")

    if not API_KEY:
        st.error("⚠️ GEMINI_API_KEY environment variable is not set. Please set your API key to use this application.")
        st.stop()

    # Settings in sidebar
    with st.sidebar:
        st.header("Summarization Settings")
        
        # Document type selection
        document_type = st.selectbox(
            "Document Type",
            ["medical", "insurance", "general"],
            format_func=lambda x: {"medical": "Medical Document", "insurance": "Insurance Document", "general": "General Document"}[x],
            help="Select the type of document to optimize summarization"
        )
        
        # Store document type in session state
        st.session_state.document_type = document_type
        
        words_limit = st.slider("Word Limit", min_value=1000, max_value=300000, value=5000, step=500,
                               help="Maximum number of words to process")
        summary_ratio = st.slider("Summary Size Ratio", min_value=0.1, max_value=0.5, value=0.33, step=0.01,
                                 help="The ratio of summary length to original document length")

        st.markdown("---")
        st.markdown("### About this app")
        st.markdown("""
        This application uses Google's Gemini AI to summarize documents and allows you to ask questions about them.

        **Features:**
        - Summarize text or PDF documents (medical, insurance, or general)
        - Ask questions about your documents
        - Adjust summarization parameters
        - Specialized handling for different document types
        """)
        st.markdown("---")
        st.markdown("Made with Streamlit and Google Gemini AI")

    # Create tabs
    tab1, tab2 = st.tabs(["Text Summarizer", "PDF Summarizer"])

    # Initialize summarizer
    summarizer = GeminiSummarizer()

    # Text tab
    with tab1:
        # Display appropriate header based on document type
        if document_type == "medical":
            st.header("Medical Text Input")
            placeholder_text = "Paste your medical document text here..."
            file_prefix = "medical"
        elif document_type == "insurance":
            st.header("insurance Text Input")
            placeholder_text = "Paste your insurance document text here (contracts, terms & conditions, policies, etc.)..."
        elif document_type == "insurance":
            st.header("insurance Text Input")
            placeholder_text = "Paste your document text here (contracts, terms & conditions, policies, etc.)..."
            file_prefix = "insurance"
        else:
            st.header("Document Text Input")
            placeholder_text = "Paste your document text here..."
            file_prefix = "document"
            
        text_input = st.text_area(f"Enter the text you want to summarize:", height=300,
                                  placeholder=placeholder_text)

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Summarize", key="summarize_text"):
                if not text_input:
                    st.error("Please enter some text to summarize.")
                else:
                    with st.spinner("Generating summary..."):
                        try:
                            word_count = len(text_input.split())
                            st.info(f"Processing {word_count} words")

                            if word_count > words_limit:
                                st.warning(f"Text is too long. Using first {words_limit} words.")

                            summary = summarizer.summarise_text(text_input, document_type, words_limit, summary_ratio)

                            st.session_state.text_content = text_input
                            st.session_state.summary = summary
                            st.session_state.show_text_chat = True
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")

        # Display summary if available
        if 'summary' in st.session_state:
            if document_type == "medical":
                st.subheader("Medical Document Summary")
            elif document_type == "insurance":
                st.subheader("Insurance Document Summary")
            else:
                st.subheader("Document Summary")
                
            st.write(st.session_state.summary)

            # Option to download summary
            summary_text = st.session_state.summary
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"{file_prefix}_summary.txt",
                mime="text/plain"
            )

            # Chat interface
            if st.session_state.get('show_text_chat', False):
                st.markdown("---")
                st.subheader("Ask Questions About This Document")
                
                if document_type == "medical":
                    st.write("Ask specific questions about diagnoses, treatments, medications, etc.:")
                elif document_type == "insurance":
                    st.write("Ask specific questions about terms, conditions, coverage, exclusions, etc.:")
                else:
                    st.write("Ask specific questions about this document:")

                # Initialize chat history if it doesn't exist
                if 'text_chat_history' not in st.session_state:
                    st.session_state.text_chat_history = []

                # Display chat history
                for message in st.session_state.text_chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                # Chat input
                user_question = st.chat_input(f"Ask a question about this {document_type} document...")
                if user_question:
                    # Add user message to chat history
                    st.session_state.text_chat_history.append({"role": "user", "content": user_question})

                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_question)

                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing document..."):
                            try:
                                response = summarizer.chat_with_text(st.session_state.text_content, user_question, document_type)
                                st.write(response)
                                # Add assistant message to chat history
                                st.session_state.text_chat_history.append({"role": "assistant", "content": response})
                            except Exception as e:
                                error_msg = f"Error generating response: {str(e)}"
                                st.error(error_msg)
                                st.session_state.text_chat_history.append({"role": "assistant", "content": error_msg})

    # PDF tab
    with tab2:
        # Display appropriate header based on document type
        if document_type == "medical":
            st.header("Medical PDF Upload")
            help_text = "Upload a medical record, research paper, or other healthcare document"
        elif document_type == "insurance":
            st.header("insurance PDF Upload")
            help_text = "Upload contracts, terms & conditions, policies, agreements, or other insurance documents"
        else:
            st.header("Document PDF Upload")
            help_text = "Upload any document you want to summarize"
            
        uploaded_file = st.file_uploader(f"Upload a {document_type} PDF document", type="pdf",
                                        help=help_text)

        if uploaded_file is not None:
            try:
                # Spinner text based on document type
                spinner_text = f"Processing {document_type.capitalize()} PDF..."
                with st.spinner(spinner_text):
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # Extract text from PDF
                    with open(pdf_path, 'rb') as file:
                        pdf_text = summarizer.extract_text_from_pdf(file)

                    # Clean up the temporary file
                    os.unlink(pdf_path)

                    word_count = len(pdf_text.split())
                    st.info(f"PDF contains approximately {word_count} words")

                    if word_count > words_limit:
                        st.warning(f"PDF is too long. Using first {words_limit} words.")

                    # Store the PDF text in session state
                    st.session_state.pdf_content = pdf_text

                    # Show a preview of the extracted text
                    with st.expander("Preview extracted text"):
                        st.text_area("Extracted text", pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text,
                                     height=200, disabled=True)

                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("Summarize PDF"):
                        with st.spinner(f"Generating {document_type} summary..."):
                            try:
                                summary = summarizer.summarise_text(pdf_text, document_type, words_limit, summary_ratio)
                                st.session_state.pdf_summary = summary
                                st.session_state.show_pdf_chat = True
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

        # Display summary if available
        if 'pdf_summary' in st.session_state:
            # Different headers based on document type
            if document_type == "medical":
                st.subheader("Medical PDF Summary")
            elif document_type == "insurance":
                st.subheader("insurance PDF Summary")
            else:
                st.subheader("Document PDF Summary")
                
            st.write(st.session_state.pdf_summary)

            # Option to download summary with appropriate filename
            pdf_summary_text = st.session_state.pdf_summary
            st.download_button(
                label="Download Summary",
                data=pdf_summary_text,
                file_name=f"{file_prefix}_pdf_summary.txt",
                mime="text/plain"
            )

            # Chat interface for PDF
            if st.session_state.get('show_pdf_chat', False):
                st.markdown("---")
                
                # Different subheaders and prompts based on document type
                if document_type == "medical":
                    st.subheader("Ask Questions About This Medical PDF")
                    st.write("Ask specific questions about diagnoses, treatments, medications, etc.:")
                elif document_type == "insurance":
                    st.subheader("Ask Questions About This Insurance PDF")
                    st.write("Ask specific questions about terms, conditions, coverage, exclusions, etc.:")
                else:
                    st.subheader("Ask Questions About This PDF")
                    st.write("Ask specific questions about this document:")

                # Initialize chat history if it doesn't exist
                if 'pdf_chat_history' not in st.session_state:
                    st.session_state.pdf_chat_history = []

                # Display chat history
                for message in st.session_state.pdf_chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                # Chat input
                user_question = st.chat_input(f"Ask a question about this {document_type} PDF...", key="pdf_chat_input")
                if user_question:
                    # Add user message to chat history
                    st.session_state.pdf_chat_history.append({"role": "user", "content": user_question})

                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_question)

                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing document..."):
                            try:
                                response = summarizer.chat_with_text(st.session_state.pdf_content, user_question, document_type)
                                st.write(response)
                                # Add assistant message to chat history
                                st.session_state.pdf_chat_history.append({"role": "assistant", "content": response})
                            except Exception as e:
                                error_msg = f"Error generating response: {str(e)}"
                                st.error(error_msg)
                                st.session_state.pdf_chat_history.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()

# Heal AI — AI-Powered Medical Claim & Insurance Assistant

Heal AI is an AI-powered platform designed to assist users with medical insurance claim processing, document summarization, policy advisory, and real-time chatbot support.

---

## 🧩 Core Features

### 📝 1. Medical Claim Document Processor
- Extracts key details from uploaded bills, reports, and prescriptions.
- Uses OCR (Tesseract) and PDF parsing (pdfplumber).
- Detects:
  - Patient name
  - Diagnosis
  - Admission/discharge dates
  - Total bill amount
  - Aadhaar number
- File uploads handled via a simple HTML interface.

### 💬 2. AI Chatbot Assistant
- Built using the Gemini API.
- Guides users through insurance and claim processes.
- Answers queries related to documentation, claim status, terms, and definitions.
- Multilingual support (e.g., English, Bengali, Nepali).
- File: `backend/ai_chatbot.py`

### 📄 3. Document Summarizer
- Summarizes key extracted data into structured and readable content.
- Useful for users and claim verification officials.
- File: `backend/doc_summarizer.py`

### 💡 4. Insurance Policy Advisor
- Recommends suitable insurance policies based on user needs (budget, health, family size).
- Compares premiums, coverage, exclusions, and more.
- Fetches the latest policies from IRDAI.
- Highlights newly released insurance schemes.
- File: `backend/ins_adv.py`

---

## 🔧 Tech Stack

- **Python** — Backend logic
- **Tesseract OCR** — Extract text from scanned documents
- **pdfplumber** — Extract structured text from PDFs
- **Regex** — Pattern matching & data cleaning
- **Flask** — Web framework (via `app.py`)
- **HTML/CSS** — User interface (`docs/`)
- **Gemini API (Google)** — AI chatbot and summarization
- **MongoDB** *(optional)* — For storing claims and user sessions

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HealAI.git
cd HealAI

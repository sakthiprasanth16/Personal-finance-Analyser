AI Finance Assistant 💰
Project summary

AI Finance Assistant is a Streamlit-based web application that transforms raw financial PDF statements into structured transactions, rich analytics, and personalized AI-powered financial advice.
​
The app handles PDF parsing, multi-layer expense categorization, spending analysis, wasteful spending detection, and generates professional PDF reports with insights and recommendations.
​

Table of Contents
Project overview

Key features

Technology stack

How it works

Usage instructions

Environment & requirements

Project structure

Deployment

Project overview
Goal: build an intelligent finance assistant that can read any bank, credit card, or UPI PDF statement and convert it into clear spending insights and actionable recommendations.
​
The project combines PDF parsing, a 3-layer categorization system, interactive analytics, and Google Gemini AI to provide personalized financial guidance and professional reports.
​

Key features
What this app does
PDF parsing

Extracts financial transactions from any bank / credit card / UPI statement PDF.

Supports wallets like Paytm, PhonePe, and GPay, as long as the PDF contains text-based transactions.
​

AI categorization

Uses Google Gemini AI plus rules to automatically categorize expenses into Essentials vs Discretionary and detailed categories.
​

Spending analysis

Generates visual charts for category-wise spending, top categories, and time-based patterns.

Detects potential wasteful spending such as subscriptions, frequent small purchases, and possible duplicates.
​

Personalized advice

Produces AI-powered financial recommendations based on the user’s transaction history and spending behavior.
​

Report generation

Builds downloadable PDF reports including summaries, charts, category breakdown, and recommendations.
​

3-layer smart categorizer
Layer 1 – Merchant database

Maps known merchants (e.g., grocery, OTT, food delivery, transport) to predefined categories for fast and accurate tagging.
​

Layer 2 – Pattern engine (regex)

Uses regex rules over transaction descriptions to detect patterns like groceries, fuel, shopping, and services.
​

Layer 3 – AI fallback (Gemini)

For unknown or ambiguous merchants, delegates final categorization to Google Gemini AI.
​

Wasteful spending detection
Identifies recurring subscriptions and memberships using periodic patterns.

Flags frequent low-value purchases that add up over time.

Highlights potential duplicate transactions based on repeated date/amount/description.

Surfaces high-value discretionary expenses for user review.
​

Technology stack
Streamlit – Web interface and user interaction.
​

Google Gemini AI – Transaction extraction assistance, categorization fallback, and personalized recommendations.
​

pdfplumber – PDF text extraction and transaction parsing.
​

Pandas / NumPy – Data cleaning, aggregation, and numerical calculations.
​

Plotly / Matplotlib – Interactive and static visualizations for spending analysis.
​

ReportLab – Generation of professional PDF reports.
​

Pillow / openpyxl – Image handling and Excel-compatible exports when needed.
​

How it works
Upload PDF

User uploads any bank, credit card, or UPI statement PDF via the Streamlit interface.
​

Extract transactions

pdfplumber reads the PDF and extracts tabular or line-based transaction data.

Custom parsing logic normalizes dates, descriptions, and amounts.
​

Categorize & tag

The 3-layer categorizer assigns merchant-level and category labels, plus Essential vs Discretionary tags.
​

Analyze spending

The app computes totals per category, time-based trends, and wasteful-spending indicators, then visualizes them.
​

Generate AI advice

Gemini uses the aggregated view to propose budgets, reduction opportunities, and savings estimates.
​

Export & report

Users can download a CSV of all transactions and a neatly formatted PDF report with charts and recommendations.
​

Usage instructions
1. Start the app
From the project root:

bash
streamlit run app.py
This usually opens the app at http://localhost:8501 in your browser.
​

2. Enter Gemini API key
On first run, the app will ask for your Google Gemini API key in a text input or sidebar field.
​

Alternatively, set an environment variable before running:

bash
export GEMINI_API_KEY="your_api_key_here"   # macOS / Linux
setx GEMINI_API_KEY "your_api_key_here"    # Windows (PowerShell)
The app reads this key to call the Gemini API for categorization and advice.
​

3. Upload a statement PDF
Click the “Choose a PDF file” button.

Select your bank / card / UPI statement PDF (e.g., Paytm, PhonePe, GPay statement).

Wait for parsing and extraction to complete; progress/spinner will be shown.
​

4. Explore tabs
Typical tabs in the UI:

Transactions

View the cleaned transaction table.

See automatically assigned category and Essential/Discretionary tags.

Check flags for subscriptions, duplicates, and high-value discretionary items.
​

Analytics

Category-wise spending pie chart.

Bar chart of top spending categories.

Time-based plots (daily / hourly trends) to see when spending spikes.
​

AI Recommendations

Generated budget suggestions per major category.

Concrete tips to reduce wasteful spending.

Estimated potential savings if suggestions are followed.
​

Download report

Button to export CSV transaction data.

Button to generate and download a PDF report with summary, charts, and AI insights.
​

Environment & requirements
Create a requirements.txt with:

text
streamlit>=1.28.0
pdfplumber>=0.10.3
pandas>=2.0.0
google-generativeai>=0.3.0
plotly>=5.17.0
reportlab>=4.0.4
matplotlib>=3.7.0
numpy>=1.24.0
pillow>=10.0.0
openpyxl>=3.1.0
These versions are compatible with modern Python 3.10+ environments and commonly used with Streamlit + Gemini + PDF tooling.
​

Recommended:

Python: 3.10 or higher.
​

Use a virtual environment (venv) to isolate dependencies.

Example setup:

bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
Project structure
A minimal repository layout:

text
ai-finance-assistant/
├── app.py             # Main Streamlit app
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore rules
└── assets/            # (Optional) logos, sample PDFs, images
Keeping the structure simple helps when deploying to Streamlit Cloud or Hugging Face Spaces.
​

Deployment
Streamlit Community Cloud
Push the project to a public GitHub repository.
​

Go to https://share.streamlit.io/ and connect your repo.
​

Set GEMINI_API_KEY in the app’s Secrets configuration.

Deploy; Streamlit will build from requirements.txt and run app.py.
​

Hugging Face Spaces (Streamlit)
Create a new Space and select Streamlit as the SDK.
​

Upload app.py, requirements.txt, and any assets.

In Settings → Secrets, add GEMINI_API_KEY with your key.

The Space will build and give you a public URL for your AI Finance Assistant.
​

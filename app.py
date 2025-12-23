import streamlit as st
# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pdfplumber
import io
import pandas as pd
import json
import re
from datetime import datetime
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')  


# ==================== TITLE & HEADER ====================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
">
    <h1 style="font-size: 3.5rem; margin: 0; color: white;">💰 AI Finance Assistant</h1>
    <p style="font-size: 1.5rem; margin: 1rem 0; color: #f0f0f0; font-weight: 300;">
        Transform Your Financial PDFs into Actionable Insights
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-size: 0.9rem;">Smart Analysis</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
            <div style="font-size: 0.9rem;">AI-Powered</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>
            <div style="font-size: 0.9rem;">Visual Reports</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💾</div>
            <div style="font-size: 0.9rem;">PDF Export</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== USER INSTRUCTIONS ====================
with st.expander("🚀 **Get Started**", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 **How It Works**
        
        1. **Upload** your financial PDF statement
        2. **AI extracts** and categorizes transactions
        3. **Get detailed** spending analysis
        4. **Receive personalized** financial advice
        5. **Download** comprehensive reports
        """)

    with col2:
        st.markdown("""
        ### 📄 **Supported Formats**
        • UPI PDF statements from Paytm, PhonePe, GPay, etc.
        • Any financial PDF with transaction data
        """)

# ==================== API KEY SETUP ====================
# Try to get API key from Hugging Face secrets first
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# If not in secrets, ask user in the main interface
if not GEMINI_API_KEY:
    st.divider()
    GEMINI_API_KEY = st.text_input(
        "**Enter your Google Gemini API Key:**",
        type="password",
        help="Get your FREE API key from https://makersuite.google.com/app/apikey"
    )

    if GEMINI_API_KEY:
        st.success("✅ API Key saved! You can now upload your PDF.")
    else:
        st.warning("⚠️ An API key is required to use this application.")
else:
    st.success("✅ API Key loaded from environment variables!")

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-card p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    .header-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    .section-title {
        color: #2d3748;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
    }
    .wasteful-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
    }
    .point-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 5px solid;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .point-card.wasteful {
        border-left-color: #ff6b6b;
        background: linear-gradient(135deg, #fff5f5 0%, #ffeaea 100%);
    }
    .point-card.savings {
        border-left-color: #ffcc00;
        background: linear-gradient(135deg, #fffaf0 0%, #fff5e6 100%);
    }
    .point-card.discretionary {
        border-left-color: #5ac8fa;
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f2ff 100%);
    }
    .highlight-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .category-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .essential-tag {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #333;
    }
    .discretionary-tag {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'transactions' not in st.session_state:
    st.session_state.transactions = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'categorized_transactions' not in st.session_state:
    st.session_state.categorized_transactions = None
if 'spending_analysis' not in st.session_state:
    st.session_state.spending_analysis = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'charts' not in st.session_state:
    st.session_state.charts = None

# ==================== 3-LAYER CATEGORIZER ====================
class SmartTransactionCategorizer:
    """3-Layer system with enhanced mapping including Essentials category"""

    def __init__(self):
        self.merchant_db = self._build_merchant_database()
        self.pattern_engine = self._build_pattern_engine()
        self.category_display_names = self._build_display_names()
        self.ai_category_mapping = self._build_ai_category_mapping()

    def _build_ai_category_mapping(self):
        """Map AI-generated categories to our standardized categories"""
        return {
            'transfer': 'transfers',
            'transfers': 'transfers',
            'upi': 'transfers',
            'payment': 'transfers',
            'shopping': 'shopping',
            'food & dining': 'food & dining',
            'food': 'food & dining',
            'groceries': 'essentials',
            'grocery': 'essentials',
            'essentials': 'essentials',
            'essential': 'essentials',
            'entertainment': 'entertainment',
            'movies': 'entertainment',
            'subscription': 'subscription',
            'transport': 'transport',
            'bills & utilities': 'bills & utilities',
            'banking & finance': 'banking & finance',
            'healthcare': 'healthcare',
            'education': 'education',
            'personal care': 'personal care',
            'income': 'income',
            'housing': 'housing',
            'donations': 'donations',
            'small purchases': 'small purchases',
            'other': 'other'
        }

    def _build_merchant_database(self):
        """Layer 1: Merchant database with essentials"""
        return {
            # Essentials - Groceries
            'bigbasket': 'essentials', 'blinkit': 'essentials', 'grofers': 'essentials',
            'dunzo': 'essentials', 'zepto': 'essentials', 'more': 'essentials',
            'dmart': 'essentials', 'reliance fresh': 'essentials', 'big bazaar': 'essentials',
            'spencer': 'essentials', 'food bazaar': 'essentials', 'hypermarket': 'essentials',
            'supermarket': 'essentials', 'kirana': 'essentials', 'medical store': 'essentials',
            'pharmacy': 'essentials', 'chemist': 'essentials', 'apollo': 'essentials',
            'medplus': 'essentials',

            # Subscriptions
            'netflix': 'subscription', 'prime video': 'subscription', 'hotstar': 'subscription',
            'disney+': 'subscription', 'spotify': 'subscription', 'amazon prime': 'subscription',
            'youtube premium': 'subscription', 'apple music': 'subscription', 'sonyliv': 'subscription',
            'zee5': 'subscription', 'jiocinema': 'subscription',

            # Shopping
            'amazon': 'shopping', 'flipkart': 'shopping', 'myntra': 'shopping',
            'nykaa': 'shopping', 'ajio': 'shopping', 'snapdeal': 'shopping',

            # Food & Dining (Restaurants)
            'zomato': 'food & dining', 'swiggy': 'food & dining', 'dominos': 'food & dining',
            'pizza hut': 'food & dining', 'kfc': 'food & dining', 'mcdonald': 'food & dining',
            'starbucks': 'food & dining', 'cafe': 'food & dining', 'restaurant': 'food & dining',
            'barista': 'food & dining', 'coffee day': 'food & dining',

            # Transport
            'uber': 'transport', 'ola': 'transport', 'rapido': 'transport',
            'auto': 'transport', 'ola money': 'transport', 'uber eats': 'food & dining',

            # Bills & Utilities
            'airtel': 'bills & utilities', 'jio': 'bills & utilities', 'bses': 'bills & utilities',
            'msedcl': 'bills & utilities', 'torrent power': 'bills & utilities', 'gas': 'bills & utilities',
            'water': 'bills & utilities', 'electricity': 'bills & utilities', 'broadband': 'bills & utilities',
            'wifi': 'bills & utilities',

            # Banking & Finance
            'atm': 'banking & finance', 'credit card': 'banking & finance', 'sbi': 'banking & finance',
            'hdfc': 'banking & finance', 'icici': 'banking & finance', 'axis': 'banking & finance',

            # Healthcare
            'pharmeasy': 'healthcare', '1mg': 'healthcare', 'netmeds': 'healthcare',

            # Education
            'byjus': 'education', 'unacademy': 'education', 'coursera': 'education',
            'udemy': 'education',

            # Entertainment
            'bookmyshow': 'entertainment', 'pvr': 'entertainment', 'inox': 'entertainment',
            'cinema': 'entertainment',

            # Personal Care
            'lakme': 'personal care', 'cult.fit': 'personal care', 'talwalkars': 'personal care',
            'haircut': 'personal care', 'salon': 'personal care', 'spa': 'personal care',

            # Income
            'salary': 'income', 'interest': 'income', 'dividend': 'income',

            # Housing
            'rent': 'housing', 'maintenance': 'housing', 'society': 'housing',

            # Transfers
            'upi': 'transfers', 'paytm': 'transfers', 'google pay': 'transfers',
            'phonepe': 'transfers', 'imps': 'transfers', 'neft': 'transfers',
            'rtgs': 'transfers',
        }

    def _build_pattern_engine(self):
        """Layer 2: Pattern engine with essentials"""
        return {
            'essentials': [
                r'\b(grocer|vegetable|fruit|milk|bread|rice|wheat|atta|pulse|dal|oil|salt|sugar|spice|tea|coffee|snack|biscuit|cookie)\b',
                r'\b(medical|medicine|tablet|syrup|injection|doctor|checkup|test|lab|diagnostic|hospital|clinic|pharmacy)\b',
                r'\b(kirana|store|supermarket|hypermarket|mart|bazaar|fresh|market|wholesale|retail)\b',
                r'\b(milk|dairy|egg|chicken|meat|fish|paneer|curd|yogurt|butter|ghee)\b',
                r'\b(rice|wheat|atta|flour|pulse|lentil|bean|legume|cereal|grain)\b',
            ],
            'subscription': [r'\b(subscription|membership|renewal|auto[\s-]?debit|recurring|prime|pro|premium|ott|streaming)\b'],
            'shopping': [r'\b(purchase|buy|order|shop|shopping|retail|store|mall|market|apparel|clothing|electronics|fashion|accessory|watch|bag)\b'],
            'food & dining': [r'\b(food|dining|restaurant|cafe|eatery|meal|breakfast|lunch|dinner|takeaway|delivery|fast[\s-]?food|buffet)\b'],
            'transport': [r'\b(transport|travel|commute|ride|cab|taxi|fuel|petrol|diesel|toll|metro|bus|train|flight|auto|rickshaw)\b'],
            'bills & utilities': [r'\b(bill|invoice|electricity|water|gas|mobile|internet|broadband|wifi|recharge|utility|maintenance|security)\b'],
            'banking & finance': [r'\b(atm|bank|card|loan|emi|investment|insurance|mutual fund|fd|finance|banking|charges|fee)\b'],
            'healthcare': [r'\b(medical|hospital|doctor|medicine|pharmacy|drug|health|clinic|diagnostic|therapy|treatment)\b'],
            'education': [r'\b(education|school|college|tuition|course|class|training|learning|book|stationery|pen|pencil|notebook)\b'],
            'entertainment': [r'\b(entertainment|movie|cinema|film|show|game|gaming|concert|theatre|music|streaming|ott|series)\b'],
            'personal care': [r'\b(salon|gym|fitness|beauty|spa|massage|haircut|personal care|wellness|cosmetic|cream|shampoo|soap)\b'],
            'income': [r'\b(salary|income|credit|deposit|interest|dividend|refund|bonus|reward|earning|paycheck|stipend)\b'],
            'housing': [r'\b(rent|house|housing|maintenance|society|flat|apartment|property|real estate|mortgage|emi)\b'],
            'transfers': [r'\b(transfer|upi|imps|neft|rtgs|send|sent|received|pay|payment|fund transfer|money transfer)\b'],
            'donations': [r'\b(donation|charity|contribution|fundraising|ngo|non-profit|help|relief|support|aid)\b'],
        }

    def _build_display_names(self):
        """Map internal categories to display names"""
        return {
            'essentials': 'Essentials',
            'subscription': 'Subscriptions',
            'shopping': 'Shopping',
            'food & dining': 'Food & Dining',
            'transport': 'Transport',
            'bills & utilities': 'Bills & Utilities',
            'banking & finance': 'Banking & Finance',
            'healthcare': 'Healthcare',
            'education': 'Education',
            'entertainment': 'Entertainment',
            'personal care': 'Personal Care',
            'income': 'Income',
            'housing': 'Housing',
            'transfers': 'Transfers',
            'donations': 'Donations',
            'small purchases': 'Small Purchases',
            'other': 'Other'
        }

    def map_ai_category(self, ai_category):
        """Map AI-generated category to our standardized category"""
        if not ai_category:
            return 'other'

        ai_category_lower = ai_category.lower().strip()

        # First check exact matches
        if ai_category_lower in self.ai_category_mapping:
            return self.ai_category_mapping[ai_category_lower]

        # Check partial matches
        for ai_key, mapped_category in self.ai_category_mapping.items():
            if ai_key in ai_category_lower:
                return mapped_category

        return 'other'

    def format_category_for_display(self, category):
        """Convert internal category to display name"""
        return self.category_display_names.get(category.lower(), category.title())

    def is_essential_category(self, category):
        """Check if a category is essential - STRICT VERSION"""
        cat_lower = category.lower()

        # ONLY these are truly essential:
        essential_list = ['essentials', 'groceries', 'housing', 'rent']

        # Also include bills/utilities
        if 'bill' in cat_lower or 'utility' in cat_lower:
            return True

        # Also include healthcare
        if 'health' in cat_lower or 'medical' in cat_lower:
            return True

        # Check if in essential list
        for essential_word in essential_list:
            if essential_word in cat_lower:
                return True

        # EVERYTHING ELSE is discretionary
        return False

    def categorize_transaction(self, description, amount, txn_type, ai_category=None):
        """3-layer categorization with AI category integration"""
        # Use AI category if available
        if ai_category:
            mapped_category = self.map_ai_category(ai_category)
            if mapped_category != 'other':
                return mapped_category

        # Convert to lowercase for matching
        text = description.lower()

        # Layer 1: Merchant database
        for merchant, category in self.merchant_db.items():
            if merchant in text:
                return category

        # Layer 2: Pattern matching
        for category, patterns in self.pattern_engine.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return category

        # Layer 3: Amount-based rules
        if txn_type.lower() in ['income', 'credit']:
            return 'income'
        elif amount < 500:
            return 'small purchases'

        return 'other'

    def analyze_spending_patterns(self, categorized_data):
        """Enhanced analysis of categorized spending patterns"""
        analysis = {
            'category_totals': {},
            'subscriptions': [],
            'small_purchases': 0,
            'top_categories': [],
            'total_income': 0,
            'total_expenses': 0,
            'monthly_income': 0,
            'monthly_expenses': 0,
            'time_based_trends': {},
            'wasteful_transactions': [],
            'category_summary': {},
            'transaction_count_by_hour': {},
            'peak_spending_hours': [],
            'essential_spending': 0,
            'discretionary_spending': 0,
            'essential_categories': [],
            'discretionary_categories': [],
            'small_purchase_details': {'total': 0, 'count': 0, 'transactions': []},
            'duplicate_transactions': [],
            'high_value_discretionary': []
        }

        if not categorized_data:
            return analysis

        df = pd.DataFrame(categorized_data)

        # Calculate total income and expenses
        income_mask = df['type'].str.lower().isin(['income', 'credit'])
        expense_mask = df['type'].str.lower().isin(['debit', 'expense'])

        analysis['total_income'] = df[income_mask]['amount'].sum()
        analysis['total_expenses'] = df[expense_mask]['amount'].sum()

        # Monthly estimates
        num_months = max(1, len(df) / 30)
        analysis['monthly_income'] = analysis['total_income'] / num_months
        analysis['monthly_expenses'] = analysis['total_expenses'] / num_months

        if 'Smart_Category' in df.columns and 'amount' in df.columns:
            expense_df = df[expense_mask]
            if not expense_df.empty:
                # Calculate category totals
                category_totals = expense_df.groupby('Smart_Category')['amount'].sum()
                analysis['category_totals'] = category_totals.to_dict()

                # Calculate essential vs discretionary spending
                for category, amount in category_totals.items():
                    cat_lower = category.lower()

                    # Essential categories
                    if self.is_essential_category(category):
                        analysis['essential_spending'] += amount
                        if category not in analysis['essential_categories']:
                            analysis['essential_categories'].append(category)
                    else:
                        # Only count as discretionary if not income or transfers
                        if cat_lower not in ['income', 'transfers']:
                            analysis['discretionary_spending'] += amount
                            if category not in analysis['discretionary_categories']:
                                analysis['discretionary_categories'].append(category)

                            # Track high-value discretionary transactions (>₹2000)
                            high_value_txns = expense_df[
                                (expense_df['Smart_Category'] == category) &
                                (expense_df['amount'] > 2000)
                            ]
                            if not high_value_txns.empty:
                                for _, row in high_value_txns.iterrows():
                                    analysis['high_value_discretionary'].append({
                                        'description': row['description'],
                                        'amount': row['amount'],
                                        'category': category,
                                        'date': row.get('date', '')
                                    })

                # Top 5 categories
                top_cats = category_totals.nlargest(5)
                analysis['top_categories'] = [
                    {'category': self.format_category_for_display(cat), 'amount': float(amt), 'internal_category': cat}
                    for cat, amt in top_cats.items()
                ]

        # Find subscriptions
        subscriptions = df[
            (df['Smart_Category'].str.lower() == 'subscription') &
            (expense_mask)
        ]
        if not subscriptions.empty:
            analysis['subscriptions'] = [
                {
                    'name': str(row['description'])[:50],
                    'monthly_cost': float(row['amount']),
                    'savings_potential': float(row['amount']) * 0.3,
                    'date': row.get('date', '')
                }
                for _, row in subscriptions.iterrows()
            ]

        # Small purchases analysis with details
        small_purchases = df[
            (df['amount'] < 500) &
            (expense_mask) &
            (df['Smart_Category'].str.lower() != 'subscription')
        ]
        if not small_purchases.empty:
            analysis['small_purchases'] = small_purchases['amount'].sum()
            analysis['small_purchase_details']['total'] = small_purchases['amount'].sum()
            analysis['small_purchase_details']['count'] = len(small_purchases)
            # Store top 10 small purchases for analysis
            analysis['small_purchase_details']['transactions'] = small_purchases.nlargest(10, 'amount').to_dict('records')

        # Find duplicate transactions
        expense_df = df[expense_mask]
        if not expense_df.empty:
            # Group by description and amount (similar transactions)
            grouped = expense_df.groupby(['description', 'amount']).size().reset_index(name='count')
            duplicates = grouped[grouped['count'] > 1]
            if not duplicates.empty:
                for _, row in duplicates.iterrows():
                    dup_txns = expense_df[
                        (expense_df['description'] == row['description']) &
                        (expense_df['amount'] == row['amount'])
                    ]
                    if len(dup_txns) > 1:
                        analysis['duplicate_transactions'].append({
                            'description': row['description'],
                            'amount': row['amount'],
                            'count': row['count'],
                            'dates': dup_txns['date'].tolist()[:3]
                        })

        # Time-based analysis
        if 'date' in df.columns and 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.day_name()

                # Peak spending hours
                expense_df = df[expense_mask]
                if not expense_df.empty:
                    expense_hours = expense_df.groupby('hour')['amount'].sum()
                    if not expense_hours.empty:
                        analysis['peak_spending_hours'] = expense_hours.nlargest(3).index.tolist()

                analysis['time_based_trends'] = {
                    'daily_patterns': df.groupby('day_of_week')['amount'].sum().to_dict(),
                    'hourly_patterns': expense_hours.to_dict() if 'expense_hours' in locals() else {}
                }
            except:
                pass

        return analysis

# ==================== EXTRACTION FUNCTIONS ====================
def extract_text_from_pdf(uploaded_file):
    """Extract all text from PDF using pdfplumber"""
    try:
        all_text = ""
        uploaded_file.seek(0)

        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"

        return all_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_transactions_with_ai(text, api_key):
    """Use Gemini AI to extract structured transactions from any PDF format"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""Extract ALL financial transactions from this document:

{text[:10000]}

Return as JSON array with these fields for each transaction:
- date (format: YYYY-MM-DD)
- time (format: HH:MM:SS, use 12:00:00 if not found)
- description (what the transaction is for)
- amount (positive number, remove currency symbols)
- type ("income" or "expense")
- category (infer from description: Shopping, Food, Transfer, Entertainment, etc.)

Return ONLY valid JSON, no other text."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            json_str = re.sub(r'```(?:json)?', '', json_str).strip()
            return json.loads(json_str)

        return []

    except Exception as e:
        st.error(f"AI extraction error: {str(e)}")
        return []

def validate_transactions(transactions):
    """Validate and clean the AI-extracted transactions"""
    if not transactions:
        return []

    validated = []

    for txn in transactions:
        validated_txn = {
            'date': txn.get('date', ''),
            'time': txn.get('time', '12:00:00'),
            'description': txn.get('description', ''),
            'amount': txn.get('amount', 0),
            'type': txn.get('type', 'unknown'),
            'category': txn.get('category', 'Other')
        }

        # Clean amount
        if isinstance(validated_txn['amount'], str):
            amount_str = re.sub(r'[₹$,¥]', '', validated_txn['amount']).strip()
            try:
                validated_txn['amount'] = float(amount_str)
            except:
                validated_txn['amount'] = 0

        # Clean time
        time_str = validated_txn['time']
        if time_str and isinstance(time_str, str):
            try:
                if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                    time_obj = datetime.strptime(time_str, '%I:%M %p')
                    validated_txn['time'] = time_obj.strftime('%H:%M:%S')
                elif ':' in time_str:
                    if time_str.count(':') == 1:
                        validated_txn['time'] = time_str + ':00'
            except:
                validated_txn['time'] = '12:00:00'

        # Ensure type is valid
        if validated_txn['type'].lower() not in ['income', 'expense', 'credit', 'debit']:
            validated_txn['type'] = 'expense'

        validated.append(validated_txn)

    return validated

# ==================== VISUALIZATIONS ====================
def create_charts(categorized_data, spending_analysis, categorizer):
    """Create comprehensive visualizations"""
    charts = {}

    if not categorized_data:
        return charts

    df = pd.DataFrame(categorized_data)

    # 1. Spending by Category Pie Chart
    if 'Smart_Category' in df.columns and 'amount' in df.columns:
        expense_df = df[df['type'].str.lower().isin(['expense', 'debit'])]

        if not expense_df.empty:
            # Format categories for display
            expense_df['Category_Display'] = expense_df['Smart_Category'].apply(
                lambda x: categorizer.format_category_for_display(x)
            )

            # Group by category
            category_totals = expense_df.groupby('Category_Display')['amount'].sum().reset_index()

            if len(category_totals) > 0:
                # Create pie chart
                fig = px.pie(
                    category_totals,
                    values='amount',
                    names='Category_Display',
                    title="",  
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                # visibility
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont=dict(size=12, color='black'),
                    marker=dict(line=dict(color='white', width=2))
                )

                fig.update_layout(
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='black', size=12)
                )

                charts['spending_categories'] = fig

    # 2. Top Categories Bar Chart
    top_cats = spending_analysis.get('top_categories', [])
    if top_cats:
        top_df = pd.DataFrame(top_cats)
        fig = px.bar(
            top_df,
            x='category',
            y='amount',
            title="",  
            color='amount',
            color_continuous_scale='Viridis',
            text='amount'
        )
        fig.update_traces(
            texttemplate='₹%{y:,.0f}',
            textposition='outside',
            marker=dict(line=dict(color='black', width=1))
        )
        fig.update_layout(
            height=350,
            xaxis_title="",
            yaxis_title="Amount (₹)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black', size=12),
            margin=dict(t=0)  
        )
        fig.update_yaxes(tickprefix='₹')
        charts['top_categories'] = fig

    # 3. Time-based trends
    if 'date' in df.columns and 'time' in df.columns:
        try:
            # Hourly spending
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            expense_df = df[df['type'].str.lower().isin(['expense', 'debit'])].copy()
            expense_df['hour'] = expense_df['datetime'].dt.hour

            hourly_spending = expense_df.groupby('hour')['amount'].sum().reset_index()

            if not hourly_spending.empty:
                fig = px.line(
                    hourly_spending,
                    x='hour',
                    y='amount',
                    title="",  
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(
                    line=dict(width=3, color='#FF6B6B'),
                    marker=dict(size=8, color='#4ECDC4')
                )
                fig.update_layout(
                    height=300,
                    xaxis_title="Hour of Day",
                    yaxis_title="Total Amount (₹)",
                    xaxis=dict(tickmode='linear', dtick=2),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0)  
                )
                fig.update_yaxes(tickprefix='₹')
                charts['hourly_trend'] = fig

            # Daily spending pattern
            df['Day'] = df['datetime'].dt.day_name()
            daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_spending = df[df['type'].str.lower().isin(['expense', 'debit'])].groupby('Day')['amount'].sum().reindex(daily_order).reset_index()

            if not daily_spending.empty:
                fig = px.bar(
                    daily_spending,
                    x='Day',
                    y='amount',
                    title="",  \
                    color='Day',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_traces(
                    texttemplate='₹%{y:,.0f}',
                    textposition='outside',
                    marker=dict(line=dict(color='black', width=1))
                )
                fig.update_layout(
                    height=300,
                    xaxis_title="Day of Week",
                    yaxis_title="Amount (₹)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(t=0)  
                )
                fig.update_yaxes(tickprefix='₹')
                charts['daily_trend'] = fig

        except Exception as e:
            pass  

    return charts

# ==================== AI RECOMMENDATIONS ====================
def generate_ai_recommendations(categorized_data, spending_analysis, api_key):
    """AI generates personalized recommendations based on actual spending patterns"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        df = pd.DataFrame(categorized_data)

        # Calculate key metrics
        total_income = spending_analysis.get('total_income', 0)
        total_expenses = spending_analysis.get('total_expenses', 0)
        net_balance = total_income - total_expenses
        savings_rate = (net_balance / total_income * 100) if total_income > 0 else 0

        # Monthly estimates
        monthly_income = spending_analysis.get('monthly_income', 0)
        monthly_expenses = spending_analysis.get('monthly_expenses', 0)

        # Essential vs Discretionary spending
        essential_spending = spending_analysis.get('essential_spending', 0)
        discretionary_spending = spending_analysis.get('discretionary_spending', 0)
        essential_percentage = (essential_spending / monthly_expenses * 100) if monthly_expenses > 0 else 0

        # Prepare top categories with details
        top_categories = []
        for cat in spending_analysis.get('top_categories', []):
            is_essential = spending_analysis.get('categorizer', SmartTransactionCategorizer()).is_essential_category(cat.get('internal_category', ''))
            category_type = "Essential" if is_essential else "Discretionary"
            top_categories.append({
                'name': cat['category'],
                'amount': cat['amount'],
                'type': category_type
            })

        # Subscriptions
        subscriptions = spending_analysis.get('subscriptions', [])
        subscription_names = [sub['name'][:30] for sub in subscriptions[:5]]
        subscription_cost = sum([sub['monthly_cost'] for sub in subscriptions])

        # Small purchases
        small_purchases = spending_analysis.get('small_purchases', 0)
        small_purchase_count = spending_analysis.get('small_purchase_details', {}).get('count', 0)

        # Transaction patterns
        total_transactions = len(df)
        income_transactions = len(df[df['type'].str.lower().isin(['income', 'credit'])])
        expense_transactions = len(df[df['type'].str.lower().isin(['expense', 'debit'])])

        # prompt for AI
        prompt = f"""You are a skilled financial advisor analyzing this person's spending data.
Provide PERSONALIZED recommendations based on THEIR SPECIFIC financial situation.

FINANCIAL PROFILE:
• Monthly Income: ₹{monthly_income:,.2f}
• Monthly Expenses: ₹{monthly_expenses:,.2f}
• Net Monthly Balance: ₹{(monthly_income - monthly_expenses):,.2f}
• Savings Rate: {savings_rate:.1f}%

SPENDING PATTERNS:
• Essential Spending: ₹{essential_spending:,.2f} ({essential_percentage:.1f}% of expenses)
• Discretionary Spending: ₹{discretionary_spending:,.2f}
• Target: 50-60% essential, 40-50% discretionary

TOP SPENDING CATEGORIES:
{chr(10).join([f"• {cat['name']}: ₹{cat['amount']:,.2f} ({cat['type']})" for cat in top_categories[:5]])}

RECURRING EXPENSES:
• Active Subscriptions: {len(subscriptions)} services (₹{subscription_cost:,.2f}/month)
• Small Purchases (<₹500): ₹{small_purchases:,.2f} across {small_purchase_count} transactions

TRANSACTION ANALYSIS:
• Total Transactions: {total_transactions}
• Income Transactions: {income_transactions}
• Expense Transactions: {expense_transactions}

IMPORTANT: Provide EXACTLY 3 sections with bullet points. Use ONLY this format:

SECTION 1: MONTHLY BUDGET PLANNING
• [Your first personalized point based on their ₹{monthly_income:,.0f} income]
• [Your second point about their {essential_percentage:.0f}% essential spending]
• [Your third point referencing their top category: {top_categories[0]['name'] if top_categories else 'spending'}]
• [Your fourth point about budgeting for their needs]
• [Your fifth point about specific allocations]

SECTION 2: SUGGESTIONS TO REDUCE UNNECESSARY SPENDING
• [Address their ₹{discretionary_spending:,.0f} discretionary spending]
• [Suggest reducing their {len(subscriptions)} subscriptions saving ₹{subscription_cost * 0.3:,.0f}/month]
• [Recommend cutting small purchases by 25% saving ₹{small_purchases * 0.25:,.0f}]
• [Target reducing {top_categories[1]['name'] if len(top_categories) > 1 else 'shopping'} spending]
• [Practical step to track spending]

SECTION 3: PERSONALIZED FINANCIAL ADVICE
• [Advice based on their ₹{(monthly_income - monthly_expenses):,.0f} net balance]
• [Guidance for their {savings_rate:.1f}% savings rate]
• [Short-term action for next month]
• [Long-term strategy based on income]
• [Final motivational advice]

EACH bullet point MUST:
1. Include their actual numbers (₹ amounts or %)
2. Reference their specific spending categories
3. Be actionable and specific
4. NOT be generic advice"""

        with st.spinner("🤖 AI is analyzing your spending patterns..."):
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 4000,
                    "temperature": 0.7
                }
            )

        # Parse response into sections
        response_text = response.text.strip()

        # Initialize sections
        sections = {
            'monthly_budget': [],
            'spending_reduction': [],
            'financial_advice': []
        }

        # Split by sections
        text_lower = response_text.lower()

        # Find section boundaries
        section1_start = text_lower.find("section 1")
        section2_start = text_lower.find("section 2")
        section3_start = text_lower.find("section 3")

        # Extract Section 1: MONTHLY BUDGET PLANNING
        if section1_start >= 0 and section2_start > section1_start:
            section1_text = response_text[section1_start:section2_start]
        else:
            # Fallback: try to find by content
            section1_text = ""
            for line in response_text.split('\n'):
                if 'monthly budget' in line.lower() or 'budget planning' in line.lower():
                    section1_text = line

        # Extract Section 2: SUGGESTIONS TO REDUCE UNNECESSARY SPENDING
        if section2_start >= 0 and section3_start > section2_start:
            section2_text = response_text[section2_start:section3_start]
        else:
            section2_text = ""

        # Extract Section 3: PERSONALIZED FINANCIAL ADVICE
        if section3_start >= 0:
            section3_text = response_text[section3_start:]
        else:
            section3_text = ""

        # Helper function to extract bullet points from text - FIXED VERSION
        def extract_bullets(text):
            bullets = []
            if not text:
                return bullets

            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Clean the line first
                clean_line = line

                # 1. Remove common bullet symbols
                bullet_prefixes = ['•', '-', '*', '◦', '›', '▸']
                for prefix in bullet_prefixes:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()

                # 2. Remove numbered bullets (1., 2), etc.)
                if clean_line and clean_line[0].isdigit():
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)

                # 3. Skip empty lines after cleaning
                if not clean_line:
                    continue

                # 4. Skip lines that are ALL CAPS section headers
                if clean_line.isupper():
                    continue

                # 5. Check for section headers more broadly
                section_keywords = ['section', 'monthly budget', 'budget planning',
                                   'reduce unnecessary spending', 'spending reduction',
                                   'personalized financial advice', 'financial advice']

                if any(keyword in clean_line.lower() for keyword in section_keywords):
                    continue

                # 6. Add the cleaned point if it has reasonable length
                if len(clean_line) >= 5:
                    bullets.append(clean_line)

            return bullets[:5]  # Return max 5 bullets

        # Extract bullets from each section
        sections['monthly_budget'] = extract_bullets(section1_text)
        sections['spending_reduction'] = extract_bullets(section2_text)
        sections['financial_advice'] = extract_bullets(section3_text)

        # If no bullets found in parsing, try alternative parsing
        if not any(sections.values()):
            # Try simpler parsing: just look for bullet points in entire response
            all_bullets = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('•'):
                    clean_line = line[1:].strip()
                    if clean_line and len(clean_line) > 15:
                        all_bullets.append(clean_line)

            # Distribute bullets evenly among sections
            if all_bullets:
                num_bullets = len(all_bullets)
                section_size = max(1, num_bullets // 3)
                sections['monthly_budget'] = all_bullets[:section_size]
                sections['spending_reduction'] = all_bullets[section_size:section_size*2]
                sections['financial_advice'] = all_bullets[section_size*2:]

        # If still no bullets, create fallback from data (NOT GENERIC)
        if not any(sections.values()):
            # Create personalized points from user's actual data
            if monthly_income > 0:
                sections['monthly_budget'] = [
                    f"Allocate ₹{monthly_income * 0.5:,.0f} for essentials (50% of ₹{monthly_income:,.0f} income)",
                    f"Budget ₹{monthly_income * 0.3:,.0f} for discretionary spending (30%)",
                    f"Save ₹{monthly_income * 0.2:,.0f} monthly (20% of income)"
                ]

            if discretionary_spending > 0:
                sections['spending_reduction'] = [
                    f"Reduce discretionary spending from ₹{discretionary_spending:,.0f} to ₹{discretionary_spending * 0.8:,.0f} (20% cut)",
                    f"Review {top_categories[0]['name'] if top_categories else 'top'} category for savings opportunities",
                    "Track daily spending for 2 weeks to identify patterns"
                ]

            sections['financial_advice'] = [
                f"Aim to save ₹{monthly_income * 0.2:,.0f} monthly from your ₹{monthly_income:,.0f} income",
                "Review expenses weekly to stay on track",
                "Set specific financial goals for motivation"
            ]

        return sections

    except Exception as e:
        # Create basic personalized guidance from available data
        monthly_income = spending_analysis.get('monthly_income', 0)
        monthly_expenses = spending_analysis.get('monthly_expenses', 0)
        discretionary_spending = spending_analysis.get('discretionary_spending', 0)

        # Build personalized guidance from actual data
        guidance = {
            'monthly_budget': [],
            'spending_reduction': [],
            'financial_advice': []
        }

        if monthly_income > 0:
            guidance['monthly_budget'].append(f"Based on your ₹{monthly_income:,.0f} income, allocate 50% (₹{monthly_income * 0.5:,.0f}) to essentials")
            guidance['monthly_budget'].append(f"Set aside 20% (₹{monthly_income * 0.2:,.0f}) for savings from your monthly income")

        if discretionary_spending > 0:
            guidance['spending_reduction'].append(f"Your ₹{discretionary_spending:,.0f} discretionary spending could be reduced by 20% (₹{discretionary_spending * 0.2:,.0f})")
            guidance['spending_reduction'].append("Review your top spending categories for reduction opportunities")

        guidance['financial_advice'].append(f"Aim to save at least ₹{monthly_income * 0.15:,.0f} monthly from your ₹{monthly_income:,.0f} income")
        guidance['financial_advice'].append("Track all expenses for 30 days to understand your spending patterns")

        return guidance

# ==================== WASTEFUL SPENDING ANALYSIS ====================
def analyze_wasteful_spending(categorized_data, spending_analysis, api_key):
    """Analyze wasteful spending with detailed, specific findings"""
    try:
        # Get detailed data from spending analysis
        df = pd.DataFrame(categorized_data)
        expense_df = df[df['type'].str.lower().isin(['expense', 'debit'])]

        if expense_df.empty:
            return {
                'wasteful_patterns': [],
                'savings_potential': [],
                'actionable_steps': []
            }

        # Extract specific wasteful patterns
        wasteful_patterns = []
        savings_potential = []
        actionable_steps = []

        # 1. Analyze Small Purchases
        small_purchase_details = spending_analysis.get('small_purchase_details', {})
        small_total = small_purchase_details.get('total', 0)
        small_count = small_purchase_details.get('count', 0)

        if small_count > 0:
            avg_small = small_total / small_count if small_count > 0 else 0
            if small_count > 20:
                wasteful_patterns.append(f"Too many small purchases: {small_count} transactions under ₹500 totaling ₹{small_total:,.2f} (Average: ₹{avg_small:,.0f})")
                savings_potential.append(f"Reduce small purchases by 25%: Save ₹{small_total * 0.25:,.0f}/month")
                actionable_steps.append(f"Set daily limit of ₹300 for small purchases and track them separately")

        # 2. Analyze Subscriptions
        subscriptions = spending_analysis.get('subscriptions', [])
        if subscriptions:
            sub_cost = sum([sub.get('monthly_cost', 0) for sub in subscriptions])
            wasteful_patterns.append(f"Multiple subscriptions: {len(subscriptions)} services costing ₹{sub_cost:,.2f}/month")

            streaming_services = [s for s in subscriptions if any(word in str(s.get('name', '')).lower() for word in ['netflix', 'prime', 'hotstar', 'disney', 'sony', 'zee'])]
            if len(streaming_services) > 2:
                wasteful_patterns.append(f"Multiple streaming services: {len(streaming_services)} platforms (consider consolidating)")

            savings_potential.append(f"Cancel 1-2 unused subscriptions: Save ₹{sub_cost * 0.3:,.0f}/month")
            actionable_steps.append("Review all subscriptions this weekend - cancel any unused for 30+ days")

        # 3. Analyze Discretionary Spending
        discretionary_spending = spending_analysis.get('discretionary_spending', 0)
        monthly_expenses = spending_analysis.get('monthly_expenses', 0)

        if monthly_expenses > 0:
            discretionary_percentage = (discretionary_spending / monthly_expenses * 100)
            if discretionary_percentage > 50:
                wasteful_patterns.append(f"High discretionary spending: {discretionary_percentage:.0f}% of total expenses (Target: ≤40%)")
                savings_potential.append(f"Reduce discretionary by 20%: Save ₹{discretionary_spending * 0.2:,.0f}/month")
                actionable_steps.append("Use cash envelopes for discretionary categories to control spending")

        # 4. Analyze Top Discretionary Categories
        top_categories = spending_analysis.get('top_categories', [])
        for cat in top_categories[:3]:
            internal_cat = cat.get('internal_category', '')
            categorizer = spending_analysis.get('categorizer')
            if categorizer and not categorizer.is_essential_category(internal_cat):
                amount = cat.get('amount', 0)
                wasteful_patterns.append(f"High {cat['category']} spending: ₹{amount:,.2f} monthly")
                savings_potential.append(f"Reduce {cat['category']} by 15%: Save ₹{amount * 0.15:,.0f}/month")

        # 5. Check for Duplicate Transactions
        duplicate_txns = spending_analysis.get('duplicate_transactions', [])
        if duplicate_txns:
            for dup in duplicate_txns[:2]:
                wasteful_patterns.append(f"Possible duplicate: '{dup.get('description', '')[:30]}...' appears {dup.get('count')} times")
                actionable_steps.append(f"Verify if ₹{dup.get('amount', 0):,.0f} transaction is duplicated on {', '.join(dup.get('dates', [])[:2])}")

        # 6. High Value Discretionary Transactions
        high_value_txns = spending_analysis.get('high_value_discretionary', [])
        if high_value_txns:
            high_value_total = sum([t.get('amount', 0) for t in high_value_txns])
            if len(high_value_txns) > 3:
                wasteful_patterns.append(f"Multiple high-value discretionary purchases: {len(high_value_txns)} transactions over ₹2000 totaling ₹{high_value_total:,.2f}")
                actionable_steps.append("Implement 48-hour cooling off period for purchases over ₹2000")

        # 7. Time-based patterns
        peak_hours = spending_analysis.get('peak_spending_hours', [])
        if peak_hours:
            if any(hour in [20, 21, 22, 23] for hour in peak_hours):
                wasteful_patterns.append("Late-night impulse spending: Peak spending during evening hours")
                actionable_steps.append("Avoid online shopping/apps after 8 PM to reduce impulse purchases")

        # If no specific patterns found, provide general guidance
        if not wasteful_patterns:
            wasteful_patterns = [
                "No severe wasteful patterns detected",
                "Your spending appears reasonably controlled",
                "Focus on optimizing existing discretionary categories"
            ]
            savings_potential = [
                "Potential 10-15% savings from optimizing current spending",
                "Review subscriptions and small purchases monthly",
                "Set specific savings goals for motivation"
            ]
            actionable_steps = [
                "Continue tracking all expenses for better insights",
                "Set monthly spending limits for discretionary categories",
                "Review financial progress every month"
            ]

        # Ensure at least 2 points in each section
        if wasteful_patterns and len(wasteful_patterns) < 2:
            wasteful_patterns.append("Review your spending categories for optimization opportunities")
        if savings_potential and len(savings_potential) < 2:
            savings_potential.append("Small adjustments can lead to significant monthly savings")
        if actionable_steps and len(actionable_steps) < 2:
            actionable_steps.append("Start with one small change this week and build from there")

        # Limit to 5 points each
        wasteful_patterns = wasteful_patterns[:5]
        savings_potential = savings_potential[:5]
        actionable_steps = actionable_steps[:5]

        return {
            'wasteful_patterns': wasteful_patterns,
            'savings_potential': savings_potential,
            'actionable_steps': actionable_steps,
            'subscriptions': subscriptions,
            'small_purchase_details': small_purchase_details,
            'duplicate_transactions': duplicate_txns
        }

    except Exception as e:
        return {
            'wasteful_patterns': [],
            'savings_potential': [],
            'actionable_steps': []
        }

# ==================== PDF REPORT GENERATION ====================
def create_comprehensive_pdf(categorized_data, spending_analysis, recommendations, charts_data):
    """Create comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=1  # Center
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=15,
        spaceBefore=20
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=5
    )

    # Title
    story.append(Paragraph("Financial Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
    story.append(Spacer(1, 30))

    # Executive Summary
    story.append(Paragraph("Executive Summary", section_style))

    monthly_income = spending_analysis.get('monthly_income', 0)
    monthly_expenses = spending_analysis.get('monthly_expenses', 0)
    net_balance = monthly_income - monthly_expenses

    summary_text = f"""
    <b>Monthly Income:</b> ₹{monthly_income:,.2f}<br/>
    <b>Monthly Expenses:</b> ₹{monthly_expenses:,.2f}<br/>
    <b>Net Monthly Balance:</b> ₹{net_balance:,.2f}<br/>
    <b>Total Transactions Analyzed:</b> {len(categorized_data)}<br/>
    <b>Analysis Period:</b> Based on transaction data provided
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Spending Analysis
    story.append(Paragraph("Spending Analysis", section_style))

    # Top Categories
    top_cats = spending_analysis.get('top_categories', [])
    if top_cats:
        story.append(Paragraph("Top Spending Categories:", styles['Heading3']))
        for cat in top_cats:
            story.append(Paragraph(f"• {cat['category']}: ₹{cat['amount']:,.2f}", bullet_style))
        story.append(Spacer(1, 10))

    # Category Summary
    df = pd.DataFrame(categorized_data)
    expense_df = df[df['type'].str.lower().isin(['expense', 'debit'])]
    if not expense_df.empty and 'Smart_Category' in expense_df.columns:
        story.append(Paragraph("Category-wise Breakdown:", styles['Heading3']))
        category_stats = expense_df.groupby('Smart_Category').agg({
            'amount': ['sum', 'count']
        }).round(2)

        # Create table
        table_data = [['Category', 'Amount', 'Transactions']]
        for idx, row in category_stats.iterrows():
            table_data.append([
                idx.title(),
                f"₹{row[('amount', 'sum')]:,.2f}",
                str(int(row[('amount', 'count')]))
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

    # AI Recommendations
    if recommendations:
        story.append(Paragraph("AI Recommendations", section_style))

        if recommendations.get('monthly_budget'):
            story.append(Paragraph("📋 Monthly Budget Plan:", styles['Heading3']))
            for point in recommendations['monthly_budget'][:5]:
                story.append(Paragraph(f"• {point}", bullet_style))
            story.append(Spacer(1, 10))

        if recommendations.get('spending_reduction'):
            story.append(Paragraph("📉 Spending Reduction Suggestions:", styles['Heading3']))
            for point in recommendations['spending_reduction'][:5]:
                story.append(Paragraph(f"• {point}", bullet_style))
            story.append(Spacer(1, 10))

        if recommendations.get('financial_advice'):
            story.append(Paragraph("🎯 Personalized Financial Advice:", styles['Heading3']))
            for point in recommendations['financial_advice'][:5]:
                story.append(Paragraph(f"• {point}", bullet_style))

    # Key Insights
    story.append(Spacer(1, 20))
    story.append(Paragraph("Key Financial Insights", section_style))

    insights_text = f"""
    <b>Savings Potential:</b> Based on your spending patterns, you could potentially save 15-25% of your discretionary spending.<br/>
    <b>Top Areas for Optimization:</b> Focus on your highest discretionary spending categories for maximum impact.<br/>
    <b>Next Steps:</b> Implement the recommendations above and review your progress monthly.<br/>
    """
    story.append(Paragraph(insights_text, styles['Normal']))

    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Generated by AI Finance Assistant", styles['Normal']))
    story.append(Paragraph("This report is for informational purposes only", styles['Italic']))

    # Build PDF
    doc.build(story)

    # Get PDF data
    pdf_data = buffer.getvalue()
    buffer.close()

    return pdf_data

# ==================== MAIN APP ====================
def main():
    # Main interface layout
    if GEMINI_API_KEY:
        st.divider()
        st.header("📤 Upload Your Financial Statement")

        uploaded_file = st.file_uploader(
            "Choose a PDF file (Paytm, PhonePe, GPay, etc,)",
            type="pdf"
        )

        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            if st.button("🚀 Analyze with AI", type="primary"):
                with st.spinner("📖 Extracting text from PDF..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.raw_text = raw_text

                    if raw_text:
                        st.info(f"✅ Extracted {len(raw_text)} characters")

                        # Process with AI
                        with st.spinner("🧠 AI is extracting transactions..."):
                            transactions = extract_transactions_with_ai(
                                raw_text,
                                GEMINI_API_KEY
                            )

                            if transactions:
                                # Validate and clean
                                validated_transactions = validate_transactions(transactions)
                                st.session_state.transactions = validated_transactions

                                # Create DataFrame
                                df = pd.DataFrame(validated_transactions)
                                st.session_state.df = df

                                st.success(f"✅ AI extracted {len(validated_transactions)} transactions!")

                                # Categorize transactions
                                with st.spinner("🔍 Categorizing transactions..."):
                                    categorizer = SmartTransactionCategorizer()

                                    categorized_transactions = []
                                    for txn in validated_transactions:
                                        smart_category = categorizer.categorize_transaction(
                                            txn.get('description', ''),
                                            txn.get('amount', 0),
                                            txn.get('type', ''),
                                            txn.get('category', '')
                                        )

                                        txn['Smart_Category'] = smart_category
                                        categorized_transactions.append(txn)

                                    st.session_state.categorized_transactions = categorized_transactions

                                    # Store categorizer in analysis for later use
                                    spending_analysis = categorizer.analyze_spending_patterns(categorized_transactions)
                                    spending_analysis['categorizer'] = categorizer
                                    st.session_state.spending_analysis = spending_analysis

                                # Create charts
                                with st.spinner("📊 Creating visualizations..."):
                                    charts = create_charts(
                                        categorized_transactions,
                                        spending_analysis,
                                        categorizer
                                    )
                                    st.session_state.charts = charts

                                # Generate AI recommendations
                                with st.spinner("💡 Generating recommendations..."):
                                    recommendations = generate_ai_recommendations(
                                        categorized_transactions,
                                        spending_analysis,
                                        GEMINI_API_KEY
                                    )
                                    st.session_state.recommendations = recommendations

                                st.success("✅ Analysis Complete!")

                            else:
                                st.error("❌ No transactions found in the PDF.")

    # Display results if available
    if st.session_state.categorized_transactions and st.session_state.df is not None:
        st.divider()

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📋 Transactions", "📈 Analytics", "📥 Download Report"])

        with tab1:
            display_transactions()

        with tab2:
            display_analytics()

        with tab3:
            display_download_report()

def display_transactions():
    """Display categorized transactions with colorful metrics"""
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Transaction Overview</h2>', unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.categorized_transactions)

    # Display colorful metrics above the table
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Transactions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        unique_cats = df['Smart_Category'].nunique() if 'Smart_Category' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>{unique_cats}</h3>
            <p>Categories</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        income_count = len(df[df['type'].str.lower().isin(['income', 'credit'])])
        expense_count = len(df[df['type'].str.lower().isin(['expense', 'debit'])])
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>{income_count}/{expense_count}</h3>
            <p>Income/Expense</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Display the DataFrame
    st.subheader("Categorized Transactions")

    display_df = df.copy()
    display_df.columns = [col.title().replace('_', ' ') for col in display_df.columns]

    # Format amount
    if 'Amount' in display_df.columns:
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f'₹{x:,.2f}')

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

def display_analytics():
    """Display analytics and visualizations with detailed wasteful spending"""
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Financial Analytics</h2>', unsafe_allow_html=True)

    if not st.session_state.categorized_transactions or not st.session_state.spending_analysis:
        st.info("No analytics data available")
        return

    categorizer = st.session_state.spending_analysis.get('categorizer', SmartTransactionCategorizer())

    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Spending by Category Pie Chart - Only show heading, not plot title
        if st.session_state.charts and 'spending_categories' in st.session_state.charts:
            st.subheader("📊 Spending by Category")
            st.plotly_chart(st.session_state.charts['spending_categories'], use_container_width=True)
        else:
            st.info("Not enough data for spending categories chart")
    
    with col2:
        # 2. Top Categories Bar Chart - Only show heading, not plot title
        if st.session_state.charts and 'top_categories' in st.session_state.charts:
            st.subheader("🏆 Top Spending Categories")
            st.plotly_chart(st.session_state.charts['top_categories'], use_container_width=True)
        else:
            st.info("Top categories data not available")

    # 3. Time-based Trends
    st.subheader("🕒 Time-based Trends")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.charts and 'hourly_trend' in st.session_state.charts:
            st.plotly_chart(st.session_state.charts['hourly_trend'], use_container_width=True)
        else:
            st.info("Hourly trend data not available")

    with col2:
        if st.session_state.charts and 'daily_trend' in st.session_state.charts:
            st.plotly_chart(st.session_state.charts['daily_trend'], use_container_width=True)
        else:
            st.info("Daily trend data not available")

    # 4. Category-wise Summary with Essentials highlight
    st.subheader("📋 Category-wise Summary")

    df = pd.DataFrame(st.session_state.categorized_transactions)
    expense_df = df[df['type'].str.lower().isin(['expense', 'debit'])]

    if not expense_df.empty and 'Smart_Category' in expense_df.columns:
        category_stats = expense_df.groupby('Smart_Category').agg({
            'amount': ['sum', 'count']
        }).round(2)

        # Display in a nice format with essential tags
        cols = st.columns(2)
        col_idx = 0

        for idx, row in category_stats.iterrows():
            with cols[col_idx % 2]:
                category_display = categorizer.format_category_for_display(idx)
                total_amount = row[('amount', 'sum')]
                transaction_count = int(row[('amount', 'count')])

                # Determine if essential or discretionary
                is_essential = categorizer.is_essential_category(idx)
                essential_tag = "🔵 Essential" if is_essential else "🟡 Discretionary"

                # Use different colors based on category type
                if is_essential:
                    bg_color = "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)"
                else:
                    bg_color = "linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%)"

                st.markdown(f"""
                <div style="background: {bg_color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; box-shadow: 0 3px 6px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; color: #333;">{category_display}</h4>
                    <p style="margin: 0.3rem 0; color: #555; font-size: 0.9rem;">
                        <span class="category-tag {'essential-tag' if is_essential else 'discretionary-tag'}">
                            {essential_tag}
                        </span>
                    </p>
                    <p style="margin: 0.5rem 0 0 0; color: #555;">
                        <b>Amount:</b> ₹{total_amount:,.2f}<br/>
                        <b>Transactions:</b> {transaction_count}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            col_idx += 1

    # 5. Wasteful Spending Detection
    st.markdown("---")
    st.subheader("⚠️ Wasteful Spending Detection")

    if st.session_state.spending_analysis and GEMINI_API_KEY:
        with st.spinner("🔍 Analyzing wasteful spending patterns..."):
            wasteful_analysis = analyze_wasteful_spending(
                st.session_state.categorized_transactions,
                st.session_state.spending_analysis,
                GEMINI_API_KEY
            )

        if wasteful_analysis and isinstance(wasteful_analysis, dict):
            # Display detailed wasteful patterns
            if wasteful_analysis.get('wasteful_patterns'):
                st.markdown("#### 🔴 Wasteful Patterns Identified")
                for point in wasteful_analysis.get('wasteful_patterns', []):
                    clean_point = point.replace('**', '')
                    st.markdown(f'<div class="point-card wasteful">• {clean_point}</div>', unsafe_allow_html=True)

            # Display Savings Potential
            if wasteful_analysis.get('savings_potential'):
                st.markdown("#### 💰 Monthly Savings Potential")
                for point in wasteful_analysis.get('savings_potential', []):
                    clean_point = point.replace('**', '')
                    st.markdown(f'<div class="point-card savings">• {clean_point}</div>', unsafe_allow_html=True)

            # Display Actionable Steps
            if wasteful_analysis.get('actionable_steps'):
                st.markdown("#### 📝 Actionable Recommendations")
                for point in wasteful_analysis.get('actionable_steps', []):
                    clean_point = point.replace('**', '')
                    st.markdown(f'<div class="point-card discretionary">• {clean_point}</div>', unsafe_allow_html=True)

            # Show message if no points in any section
            if not (wasteful_analysis.get('wasteful_patterns') or
                    wasteful_analysis.get('savings_potential') or
                    wasteful_analysis.get('actionable_steps')):
                st.info("No wasteful spending patterns detected. Your spending looks reasonable!")
        else:
            st.info("No wasteful spending analysis available")
    else:
        st.info("Analyze transactions first to detect wasteful spending")

    # 6. AI Recommendations
    st.markdown("---")
    st.subheader("💰 Personalized Recommendations")

    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations

        # Check if any recommendations exist
        has_recommendations = False

        # Monthly Budget Planning
        if recommendations.get('monthly_budget'):
            has_recommendations = True
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      padding: 1rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white;">○ Monthly Budget Planning</h4>
            </div>
            """, unsafe_allow_html=True)

            for point in recommendations['monthly_budget'][:4]:
                st.markdown(f'<div class="point-card">• {point}</div>', unsafe_allow_html=True)

        # Spending Reduction Suggestions
        if recommendations.get('spending_reduction'):
            has_recommendations = True
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                      padding: 1rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white;">○ Reduce Unnecessary Spending</h4>
            </div>
            """, unsafe_allow_html=True)

            for point in recommendations['spending_reduction'][:4]:
                st.markdown(f'<div class="point-card">• {point}</div>', unsafe_allow_html=True)

        # Personalized Financial Advice
        if recommendations.get('financial_advice'):
            has_recommendations = True
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                      padding: 1rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white;">○ Personalized Financial Advice</h4>
            </div>
            """, unsafe_allow_html=True)

            for point in recommendations['financial_advice'][:4]:
                st.markdown(f'<div class="point-card">• {point}</div>', unsafe_allow_html=True)

        if not has_recommendations:
            st.info("AI recommendations will appear here after analysis")

        # Add summary metrics - ONLY Monthly Savings Potential
        st.markdown("---")
        col1 = st.columns(1)[0]  # Single column for savings potential

        with col1:
            monthly_expenses = st.session_state.spending_analysis.get('monthly_expenses', 0)
            discretionary_spending = st.session_state.spending_analysis.get('discretionary_spending', 0)

            # REALITY CHECK: If discretionary is too low, adjust
            if monthly_expenses > 0 and discretionary_spending < monthly_expenses * 0.2:
                discretionary_spending = monthly_expenses * 0.3

            savings_potential = min(discretionary_spending * 0.15, monthly_expenses * 0.2) if monthly_expenses > 0 else 0

            st.markdown(f"""
            <div class="highlight-box">
                💰 Monthly Savings Potential
                <h3>₹{savings_potential:,.0f}</h3>
                <p style="font-size: 0.9rem; margin: 0;">
                    From discretionary spending optimization
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Recommendations will appear here after analysis")

def display_download_report():
    """Display download options"""
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Download Reports</h2>', unsafe_allow_html=True)

    # Download CSV
    st.subheader("📊 Transaction Data (CSV)")
    st.write("Download all categorized transactions in CSV format.")

    if st.session_state.categorized_transactions:
        csv = pd.DataFrame(st.session_state.categorized_transactions).to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="financial_transactions.csv",
            mime="text/csv",
            help="Download all transactions as CSV file"
        )

    st.divider()

    # Download PDF Report
    st.subheader("📄 Comprehensive Analysis Report (PDF)")
    st.write("Download a professional PDF report with all analytics and recommendations.")

    if (st.session_state.categorized_transactions and
        st.session_state.spending_analysis and
        st.session_state.recommendations):

        if st.button("🔄 Generate PDF Report", type="primary"):
            with st.spinner("Creating professional PDF report..."):
                # Generate PDF
                pdf_data = create_comprehensive_pdf(
                    st.session_state.categorized_transactions,
                    st.session_state.spending_analysis,
                    st.session_state.recommendations,
                    st.session_state.charts
                )

                # Show preview of what's in the PDF
                st.markdown("""
                **📋 PDF Report Contents:**
                - Executive Summary
                - Spending Analysis
                - Category Breakdown
                - AI Recommendations
                - Key Financial Insights
                """)

                # Download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name="financial_analysis_report.pdf",
                    mime="application/pdf",
                    help="Download complete analysis as PDF"
                )
    else:
        st.info("Complete the analysis first to generate PDF report")

if __name__ == "__main__":
    main()

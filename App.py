import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import re
import json
import time

# Try to import google.generativeai for AI comps (optional)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Dynamic Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# CSS for better styling
st.markdown("""
<style>
.main { padding-top: 1rem; }
.stMainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DEFAULT API KEYS - Add your Gemini API keys here for rotation
# =============================================================================
DEFAULT_API_KEYS = [
    "AIzaSyDTlvBVoi_22xGbaT9hLYKeiYwWRu0qEnY",  # Replace with your actual API key
    # Add more keys for rotation (optional)
    # "YOUR_GEMINI_API_KEY_3",  # Uncomment and add more as needed
]
# =============================================================================

# --- AI Comps Session State Initialization ---
if 'ai_comps_api_keys' not in st.session_state:
    st.session_state.ai_comps_api_keys = [k for k in DEFAULT_API_KEYS if k and not k.startswith("YOUR_")]
if 'ai_comps_results_df' not in st.session_state:
    st.session_state.ai_comps_results_df = None
if 'ai_comps_similarity_threshold' not in st.session_state:
    st.session_state.ai_comps_similarity_threshold = 50
if 'ai_comps_current_key_index' not in st.session_state:
    st.session_state.ai_comps_current_key_index = 0
if 'ai_comps_plot_mode' not in st.session_state:
    st.session_state.ai_comps_plot_mode = "Highlight Mode"  # "Highlight Mode" or "Filter Mode"

# --- AI Comps Helper Functions ---
def ai_comps_normalize_symbol(s):
    """Normalize symbol to uppercase with .NS suffix for matching."""
    s = str(s).strip().upper()
    return s if s.endswith('.NS') else s + '.NS'

def ai_comps_clean_relevance_score(score):
    """Safely converts the relevance score to a float between 0 and 100."""
    if pd.isna(score): return 0.00
    if isinstance(score, (int, float)): return float(score)
    if isinstance(score, str):
        cleaned = re.sub(r'[^\d.]', '', str(score))
        try:
            return float(cleaned) if cleaned else 0.00
        except ValueError:
            return 0.00
    return 0.00

def ai_comps_get_similarity_map():
    """Build similarity map from AI comps results if available."""
    if st.session_state.ai_comps_results_df is None:
        return {}
    
    results_df = st.session_state.ai_comps_results_df
    
    # Find the relevance score column (case-insensitive)
    score_col = None
    for col in results_df.columns:
        if col.lower().replace(' ', '_') in ['relevance_score', 'relevancescore', 'relevance score']:
            score_col = col
            break
    
    if score_col is None:
        return {}
    
    # Find the company name column
    name_col = None
    for col in results_df.columns:
        if col.lower().replace(' ', '_') in ['company_name', 'companyname', 'company name', 'symbol']:
            name_col = col
            break
    
    if name_col is None:
        return {}
    
    similarity_map = {}
    for _, row in results_df.iterrows():
        symbol = ai_comps_normalize_symbol(row[name_col])
        score = ai_comps_clean_relevance_score(row[score_col])
        similarity_map[symbol] = score
        # Also add without .NS for flexible matching
        symbol_short = symbol.replace('.NS', '')
        similarity_map[symbol_short] = score
    
    return similarity_map

def ai_comps_load_gemini_model(api_key):
    """Loads and validates the Gemini model with the given API key."""
    if not GENAI_AVAILABLE:
        raise Exception("google-generativeai package not installed. Run: pip install google-generativeai")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        # Simple test to validate the key and model
        test_response = model.generate_content("Say OK")
        if "OK" not in test_response.text:
            raise RuntimeError("Gemini model did not respond as expected.")
        return model
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini model. Check your API Key. Error: {e}")

def ai_comps_process_batch(batch_df, target_bd, model, target_company_name):
    """Processes a single batch of companies by sending them to the AI."""
    companies_data = []
    for _, row in batch_df.iterrows():
        comp_name = row.name if isinstance(row.name, str) else str(row.get("Company Name", row.name))
        comp_name = comp_name.replace('.NS', '')
        comp_bd = row.get("longName", "") or row.get("shortName", "") or "No business description available"
        companies_data.append({"name": str(comp_name), "description": str(comp_bd)})

    prompt = f"""
You are a financial analyst specializing in competitive intelligence. Your task is to analyze a list of companies and compare them to a primary target company based on their business descriptions.

**TARGET COMPANY'S NAME:** {target_company_name}
**TARGET COMPANY'S BUSINESS DESCRIPTION:**
{target_bd}

**COMPANIES TO ANALYZE (PEERS IN THE SAME INDUSTRY):**
{chr(10).join([f"{i + 1}. {comp['name']}: {comp['description']}" for i, comp in enumerate(companies_data)])}

For each company in the list, provide the following analysis:

1.  **Business Summary**: A concise 1-2 sentence summary of what the company does.
2.  **Business Model**: How the company primarily generates revenue (e.g., B2B, B2C, SaaS, advertising).
3.  **Key Products/Services**: The main products or services offered.
4.  **Relevance Score**: A numerical score from 1.00 to 100.00 indicating how similar the company's business is to the target company. A higher score means a more direct competitor. If the company being analyzed is the target company itself ({target_company_name}), its score MUST be 100.00.
5.  **Relevance Reason**: A brief 1-2 sentence explanation for the given relevance score.

**Required Response Format (Strict JSON):**
```json
{{
  "companies": [
    {{
      "company_name": "Company Name",
      "business_summary": "Clear summary of what they do.",
      "business_model": "How they make money.",
      "key_products_services": "Main products/services.",
      "relevance_score": 85.50,
      "relevance_reason": "Reason for the score, comparing to the target."
    }}
  ]
}}
```

IMPORTANT:
- The `relevance_score` MUST be a numeric value (like 85.50).
- The JSON response MUST be perfectly formatted.
- It is MANDATORY to return a JSON object for every single company provided in the input list.
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            full_response = response.text.strip()

            json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start == -1 or json_end == 0: raise ValueError("No JSON found in response")
                json_str = full_response[json_start:json_end]

            parsed_data = json.loads(json_str)
            companies_analysis = parsed_data.get('companies', [])

            batch_results = []
            for i, comp_data in enumerate(companies_data):
                original_row = batch_df.iloc[i]
                analysis = next((item for item in companies_analysis if
                                 item.get("company_name", "").lower() == comp_data["name"].lower()), {})

                result_entry = {
                    "Company Name": comp_data["name"],
                    "Industry": original_row.get("industry", "N/A"),
                    "Business Summary": analysis.get("business_summary", "N/A"),
                    "Business Model": analysis.get("business_model", "N/A"),
                    "Key Products/Services": analysis.get("key_products_services", "N/A"),
                    "Relevance Score": analysis.get("relevance_score", 0.00),
                    "Relevance Reason": analysis.get("relevance_reason", "AI did not return data for this company.")
                }
                batch_results.append(result_entry)
            return batch_results, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                error_results = [{
                    "Company Name": comp["name"], "Industry": batch_df.iloc[i].get("industry", "N/A"),
                    "Business Summary": "Processing failed", "Business Model": "Error",
                    "Key Products/Services": "Error", "Relevance Score": 0.00,
                    "Relevance Reason": f"API/Parsing Error: {str(e)}"
                } for i, comp in enumerate(companies_data)]
                return error_results, f"A batch failed after {max_retries} attempts. Error: {e}"

def ai_comps_run_analysis(df_to_process, target_bd, target_company_name, batch_size=5):
    """Runs AI comps analysis on the given dataframe with automatic key rotation."""
    api_keys = st.session_state.ai_comps_api_keys
    
    if not api_keys:
        st.error("Cannot start analysis: No API keys available. Add keys to DEFAULT_API_KEYS in code or sidebar.")
        return None
    
    progress_bar = st.progress(0, "Initializing AI analysis...")
    
    # Start from the current key index for rotation
    current_key_idx = st.session_state.ai_comps_current_key_index % len(api_keys)
    model = None
    key_attempts = 0
    
    # Try to initialize model with key rotation
    while key_attempts < len(api_keys):
        try:
            api_key = api_keys[current_key_idx]
            model = ai_comps_load_gemini_model(api_key)
            st.session_state.ai_comps_current_key_index = current_key_idx
            break
        except Exception as e:
            st.warning(f"Key {current_key_idx + 1} failed, trying next key...")
            current_key_idx = (current_key_idx + 1) % len(api_keys)
            key_attempts += 1
    
    if model is None:
        st.error("❌ All API keys failed. Please check your API keys.")
        progress_bar.empty()
        return None
    
    total_batches = (len(df_to_process) + batch_size - 1) // batch_size
    all_results = []
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df_to_process))
        batch_df = df_to_process.iloc[start_idx:end_idx]
        
        progress_bar.progress((i + 1) / total_batches, f"Processing Batch {i + 1}/{total_batches}...")
        
        # Try processing with key rotation on failure
        batch_results = None
        batch_key_attempts = 0
        
        while batch_key_attempts < len(api_keys):
            try:
                batch_results, error = ai_comps_process_batch(batch_df, target_bd, model, target_company_name)
                if error and "quota" in str(error).lower():
                    raise Exception("Quota exceeded")
                break
            except Exception as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    # Rotate to next key
                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                    st.session_state.ai_comps_current_key_index = current_key_idx
                    try:
                        model = ai_comps_load_gemini_model(api_keys[current_key_idx])
                        batch_key_attempts += 1
                        continue
                    except:
                        batch_key_attempts += 1
                        continue
                else:
                    batch_results, error = ai_comps_process_batch(batch_df, target_bd, model, target_company_name)
                    break
        
        if batch_results:
            all_results.extend(batch_results)
        
        if i < total_batches - 1:
            time.sleep(1)
    
    # Rotate key for next run
    st.session_state.ai_comps_current_key_index = (current_key_idx + 1) % len(api_keys)
    
    final_df = pd.DataFrame(all_results)
    final_df['Relevance Score'] = final_df['Relevance Score'].apply(ai_comps_clean_relevance_score).clip(0, 100)
    final_df = final_df.sort_values(by='Relevance Score', ascending=False)
    
    progress_bar.empty()
    st.success(f"✅ AI Analysis Complete! Processed {len(final_df)} companies.")
    
    return final_df

def handle_url_parameters():
    """Handle URL parameters for Excel integration"""
    navigation_info = {}
    
    try:
        query_params = st.query_params
        # Stock param (?stock=RELIANCE)
        if 'stock' in query_params:
            stock_symbol = str(query_params['stock']).upper()
            if not stock_symbol.endswith('.NS'):
                stock_symbol += '.NS'
            navigation_info['type'] = 'stock'
            navigation_info['value'] = stock_symbol
        elif 'industry' in query_params:
            industry_name = str(query_params['industry'])
            navigation_info['type'] = 'industry'
            navigation_info['value'] = industry_name
        elif 'search' in query_params:
            search_term = str(query_params['search'])
            navigation_info['type'] = 'search'
            navigation_info['value'] = search_term
        # Mode param (?mode=scatter or ?mode=concall)
        if 'mode' in query_params:
            raw_mode = str(query_params['mode']).lower()
            if raw_mode in ['scatter','charts']:
                navigation_info['mode'] = 'scatter'
            elif raw_mode in ['concall','summary']:
                navigation_info['mode'] = 'concall'
    except Exception:
        pass
    return navigation_info

def search_stocks_and_industries(df, search_term):
    """Search for stocks and industries based on search term"""
    results = {
        'stocks': [],
        'industries': [],
        'companies': []
    }
    
    search_lower = search_term.lower()
    
    # Search in stock symbols
    stock_matches = [stock for stock in df.index if search_lower in stock.lower()]
    results['stocks'].extend(stock_matches)
    
    # Search in company names
    if 'longName' in df.columns:
        name_matches = df[df['longName'].str.contains(search_term, case=False, na=False)]
        for idx, row in name_matches.iterrows():
            results['companies'].append({
                'symbol': idx,
                'name': row['longName']
            })
    
    # Search in short names
    if 'shortName' in df.columns:
        short_name_matches = df[df['shortName'].str.contains(search_term, case=False, na=False)]
        for idx, row in short_name_matches.iterrows():
            if idx not in [item['symbol'] for item in results['companies']]:
                results['companies'].append({
                    'symbol': idx,
                    'name': row.get('shortName', 'N/A')
                })
    
    # Search in industries
    if 'industry' in df.columns:
        industry_matches = df['industry'].dropna().unique()
        industry_matches = [ind for ind in industry_matches if search_lower in ind.lower()]
        results['industries'].extend(industry_matches)
    
    # Remove duplicates
    results['stocks'] = list(set(results['stocks']))
    results['industries'] = list(set(results['industries']))
    
    return results

def display_navigation_banner(navigation_info, df):
    """Process navigation info and return values without displaying anything"""
    if not navigation_info:
        return None, None, None
    
    nav_type = navigation_info.get('type')
    nav_value = navigation_info.get('value')
    
    if not nav_type or not nav_value:
        return None, None, None
    
    if nav_type == 'stock':
        if nav_value in df.index:
            industry = df.loc[nav_value].get('industry', 'Unknown Industry')
            return 'stock', nav_value, industry
        else:
            return None, None, None
    
    elif nav_type == 'industry':
        if 'industry' in df.columns:
            matching_stocks = df[df['industry'].str.contains(nav_value, case=False, na=False)]
            if len(matching_stocks) > 0:
                return 'industry', None, nav_value
        return None, None, None
    
    elif nav_type == 'search':
        search_results = search_stocks_and_industries(df, nav_value)
        # Auto-select first stock result
        if search_results['stocks']:
            selected_stock = search_results['stocks'][0]
            industry = df.loc[selected_stock].get('industry', 'Unknown')
            return 'stock', selected_stock, industry
        elif search_results['industries']:
            return 'industry', None, search_results['industries'][0]
        return None, None, None
    
    return None, None, None

@st.cache_data
def load_and_process_data():
    """Load and process stock info CSV file"""
    try:
        if os.path.exists('stocks_info.csv'):
            df = pd.read_csv('stocks_info.csv')
            # Pivot the data to get stocks as rows and fields as columns
            pivoted_df = df.pivot(index='Stock', columns='Field', values='Value')
            return pivoted_df, None
        else:
            return None, "No stock data files found. Please ensure 'stocks_info.csv' exists in the current directory."
        
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_concall_summaries():
    """Load concall / summary CSV if present. Returns (df, error)."""
    candidates = ["Summary.csv", "summary.csv", "ConcallSummary.csv"]
    for f in candidates:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                df.columns = [c.strip() for c in df.columns]
                symbol_col = next((c for c in ["Stock", "Symbol", "Ticker"] if c in df.columns), None)
                if not symbol_col:
                    return None, f"Missing symbol column in {f}."
                df[symbol_col] = df[symbol_col].astype(str).str.upper().apply(lambda s: s if s.endswith('.NS') else s + '.NS')
                return df, None
            except Exception as e:
                return None, f"Error reading {f}: {e}"
    return None, "No summary file found (Summary.csv / summary.csv / ConcallSummary.csv)."

@st.cache_data
def load_index_constituents():
    """
    Load index constituents from CSV file.
    Returns a dictionary: { index_name: [list of symbols with .NS suffix] }
    """
    candidates = ["index_constituents_long.csv", "nse_all_index_constituents.csv"]
    for f in candidates:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                df.columns = [c.strip() for c in df.columns]
                
                # Find the symbol and index columns
                symbol_col = next((c for c in ["Symbol", "symbol", "SYMBOL", "Stock", "stock"] if c in df.columns), None)
                index_col = next((c for c in ["Index", "index", "INDEX", "IndexName", "index_name"] if c in df.columns), None)
                
                if not symbol_col or not index_col:
                    continue
                
                # Build the dictionary
                index_dict = {}
                for _, row in df.iterrows():
                    idx_name = str(row[index_col]).strip()
                    symbol = str(row[symbol_col]).strip().upper()
                    
                    # Add .NS suffix if not present
                    if not symbol.endswith('.NS'):
                        symbol = symbol + '.NS'
                    
                    if idx_name not in index_dict:
                        index_dict[idx_name] = []
                    if symbol not in index_dict[idx_name]:
                        index_dict[idx_name].append(symbol)
                
                return index_dict, None
            except Exception as e:
                return None, f"Error reading {f}: {e}"
    
    return None, "No index constituents file found (index_constituents_long.csv or nse_all_index_constituents.csv)."

@st.cache_data
def get_available_industries(df):
    """Get list of available industries"""
    if 'industry' not in df.columns:
        return []
    
    industries = df['industry'].dropna().unique()
    return sorted(industries)

def get_numeric_columns(df):
    """Get list of columns that can be converted to numeric"""
    numeric_cols = []
    
    # Basic financial metrics available in stocks_info.csv
    common_metrics = [
        'trailingPE', 'forwardPE', 'priceToBook', 'returnOnEquity', 'returnOnAssets',
        'earningsGrowth', 'revenueGrowth', 'marketCap', 'currentPrice', 'targetMeanPrice',
        'debtToEquity', 'currentRatio', 'quickRatio', 'grossMargins', 'operatingMargins',
        'profitMargins', 'ebitdaMargins', 'dividendYield', 'payoutRatio', 'beta',
        'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiftyDayAverage', 'twoHundredDayAverage',
        'fiftyDayAverageChangePercent', 'twoHundredDayAverageChangePercent',
        'enterpriseToRevenue', 'enterpriseToEbitda', 'pegRatio', 'trailingEps',
        'forwardEps', 'bookValue', 'priceToSalesTrailing12Months', 'totalCash',
        'totalDebt', 'totalRevenue', 'freeCashflow', 'operatingCashflow',
        'trailingPegRatio', 'fiveYearAvgDividendYield', 'trailingAnnualDividendYield'
    ]
    
    for col in common_metrics:
        if col in df.columns:
            # Test if the column can be converted to numeric
            sample_data = df[col].dropna().head(10)
            if len(sample_data) > 0:
                try:
                    pd.to_numeric(sample_data, errors='raise')
                    numeric_cols.append(col)
                except:
                    continue
    
    return sorted(numeric_cols)

def filter_by_industry(df, industry_filter):
    """Filter stocks by industry"""
    if 'industry' not in df.columns:
        return None
    
    # Filter stocks that contain the industry keyword (case-insensitive)
    filtered_df = df[df['industry'].str.contains(industry_filter, case=False, na=False)]
    
    return filtered_df

def filter_by_symbols(df, symbols_list):
    """
    Filter stocks by a list of symbols.
    Returns a DataFrame containing only the stocks in the symbols_list.
    """
    # Ensure symbols have .NS suffix for matching
    symbols_with_ns = []
    for s in symbols_list:
        s_upper = s.upper()
        if not s_upper.endswith('.NS'):
            s_upper = s_upper + '.NS'
        symbols_with_ns.append(s_upper)
    
    # Filter the dataframe to only include stocks in the list
    filtered_df = df[df.index.isin(symbols_with_ns)]
    
    return filtered_df

def extract_metrics(df, x_col, y_col):
    """Extract metrics for plotting"""
    required_cols = [x_col, y_col, 'industry']
    
    # Add optional columns if they exist
    optional_cols = ['longName', 'shortName', 'marketCap', 'currentPrice', 
                    'fiftyDayAverageChangePercent', 'twoHundredDayAverageChangePercent']
    
    available_cols = required_cols.copy()
    for col in optional_cols:
        if col in df.columns:
            available_cols.append(col)
    
    # Remove duplicates
    available_cols = list(set(available_cols))
    
    try:
        metrics_df = df[available_cols].copy()
    except KeyError as e:
        return None, f"KeyError when accessing columns: {e}"
    
    # Convert numeric columns
    for col in [x_col, y_col]:
        if col in metrics_df.columns:
            try:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                continue
    
    # Remove rows with missing data for the main metrics
    initial_count = len(metrics_df)
    metrics_df = metrics_df.dropna(subset=[x_col, y_col])
    final_count = len(metrics_df)
    
    return metrics_df, f"Processed {final_count} stocks (removed {initial_count - final_count} with missing data)"

def create_plotly_scatter(metrics_df, industry_filter, x_col, y_col, remove_outliers=True, highlight_stock=None, autoscale=False, use_gl=True, debug=False, similarity_threshold=None, similarity_map=None):
    """Create an interactive scatter plot using Plotly (robust fallback sizes)."""
    # Get nice display names for the axes
    axis_names = {
        'trailingPE': 'Trailing P/E Ratio',
        'forwardPE': 'Forward P/E Ratio',
        'priceToBook': 'Price-to-Book Ratio',
        'returnOnEquity': 'Return on Equity (%)',
        'returnOnAssets': 'Return on Assets (%)',
        'earningsGrowth': 'Earnings Growth (%)',
        'revenueGrowth': 'Revenue Growth (%)',
        'marketCap': 'Market Cap (₹ Cr)',
        'currentPrice': 'Current Price (₹)',
        'debtToEquity': 'Debt-to-Equity Ratio',
        'currentRatio': 'Current Ratio',
        'quickRatio': 'Quick Ratio',
        'grossMargins': 'Gross Margins (%)',
        'operatingMargins': 'Operating Margins (%)',
        'profitMargins': 'Profit Margins (%)',
        'ebitdaMargins': 'EBITDA Margins (%)',
        'dividendYield': 'Dividend Yield (%)',
        'payoutRatio': 'Payout Ratio (%)',
        'beta': 'Beta',
        'pegRatio': 'PEG Ratio',
        'trailingPegRatio': 'Trailing PEG Ratio',
        'fiftyTwoWeekHigh': '52 Week High (₹)',
        'fiftyTwoWeekLow': '52 Week Low (₹)',
        'fiftyDayAverage': '50 Day Average (₹)',
        'twoHundredDayAverage': '200 Day Average (₹)',
        'fiftyDayAverageChangePercent': '50 Day Avg Change (%)',
        'twoHundredDayAverageChangePercent': '200 Day Avg Change (%)',
        'enterpriseToRevenue': 'Enterprise to Revenue',
        'enterpriseToEbitda': 'Enterprise to EBITDA',
        'trailingEps': 'Trailing EPS (₹)',
        'forwardEps': 'Forward EPS (₹)',
        'bookValue': 'Book Value (₹)',
        'priceToSalesTrailing12Months': 'Price to Sales (TTM)',
        'totalCash': 'Total Cash (₹)',
        'totalDebt': 'Total Debt (₹)',
        'totalRevenue': 'Total Revenue (₹)',
        'freeCashflow': 'Free Cash Flow (₹)',
        'operatingCashflow': 'Operating Cash Flow (₹)',
        'targetMeanPrice': 'Target Mean Price (₹)',
        'fiveYearAvgDividendYield': '5-Year Avg Dividend Yield (%)',
        'trailingAnnualDividendYield': 'Trailing Annual Dividend Yield (%)'
    }
    x_title = axis_names.get(x_col, x_col.replace('_', ' ').title())
    y_title = axis_names.get(y_col, y_col.replace('_', ' ').title())
    percentage_cols = ['returnOnEquity','returnOnAssets','earningsGrowth','revenueGrowth','grossMargins','operatingMargins','profitMargins','ebitdaMargins','dividendYield','payoutRatio','fiftyDayAverageChangePercent','twoHundredDayAverageChangePercent','fiveYearAvgDividendYield','trailingAnnualDividendYield']
    market_cap_cols = ['marketCap']
    x_multiplier = 100 if x_col in percentage_cols and metrics_df[x_col].max() <= 3 else (1/10000000 if x_col in market_cap_cols else 1)
    y_multiplier = 100 if y_col in percentage_cols and metrics_df[y_col].max() <= 3 else (1/10000000 if y_col in market_cap_cols else 1)
    if remove_outliers:
        def remove_outliers_iqr(series):
            Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
            return (series >= lower_bound) & (series <= upper_bound)
        x_filter = remove_outliers_iqr(metrics_df[x_col]); y_filter = remove_outliers_iqr(metrics_df[y_col])
        filtered_metrics = metrics_df[x_filter & y_filter]
    else:
        filtered_metrics = metrics_df.copy()
    filtered_metrics = filtered_metrics.replace([np.inf,-np.inf], np.nan).dropna(subset=[x_col,y_col])
    if len(filtered_metrics) == 0:
        return None, "No data available after filtering"
    median_x = filtered_metrics[x_col].median(); median_y = filtered_metrics[y_col].median()
    hover_text = []
    for idx, row in filtered_metrics.iterrows():
        stock_name = idx.replace('.NS',''); company_name = row.get('longName', row.get('shortName','N/A'))
        # Get market cap in crores
        mcap_value = pd.to_numeric(row.get('marketCap', None), errors='coerce')
        if pd.notna(mcap_value):
            mcap_crores = mcap_value / 10000000  # Convert to crores
            if mcap_crores >= 1000:
                mcap_str = f"{mcap_crores/1000:.2f}K Cr"
            else:
                mcap_str = f"{mcap_crores:.2f} Cr"
        else:
            mcap_str = "N/A"
        
        # Add similarity score to hover if available
        similarity_str = ""
        if similarity_map is not None:
            idx_normalized = ai_comps_normalize_symbol(idx)
            idx_short = idx.replace('.NS', '').upper()
            score = similarity_map.get(idx_normalized) or similarity_map.get(idx_short) or similarity_map.get(idx)
            if score is not None:
                similarity_str = f"<br>Similarity: {score:.1f}%"
            else:
                similarity_str = "<br>Similarity: N/A"
        
        hover_info = f"<b>{stock_name}</b><br>Company: {company_name}<br>Market Cap: ₹{mcap_str}<br>{x_title}: {row[x_col]*x_multiplier:.2f}<br>{y_title}: {row[y_col]*y_multiplier:.2f}{similarity_str}"; hover_text.append(hover_info)
    fig = go.Figure()
    # Robust marker sizes based on market cap (use log scale for better differentiation)
    if 'marketCap' in filtered_metrics.columns:
        mc = pd.to_numeric(filtered_metrics['marketCap'], errors='coerce')
        mc_valid = mc.dropna()
        if mc_valid.empty:
            base_sizes = pd.Series([12]*len(filtered_metrics), index=filtered_metrics.index)
        else:
            # Use log scale for market cap to get better size differentiation
            mc_log = np.log10(mc.fillna(mc_valid.median()).clip(lower=1))
            mc_log_valid = np.log10(mc_valid.clip(lower=1))
            log_min = mc_log_valid.min()
            log_max = mc_log_valid.max()
            log_range = log_max - log_min
            if log_range == 0:
                mc_norm = pd.Series([0.5]*len(filtered_metrics), index=filtered_metrics.index)
            else:
                mc_norm = (mc_log - log_min) / log_range
            mc_norm = mc_norm.clip(0, 1)
            # Size range from 5 (smallest/low mcap) to 28 (largest/high mcap) for more visible differentiation
            base_sizes = (3 + mc_norm * 17).fillna(12)
    else:
        base_sizes = pd.Series([12]*len(filtered_metrics), index=filtered_metrics.index)
    # Replace any remaining invalid sizes
    base_sizes = base_sizes.fillna(12)
    
    # Color logic with similarity highlighting
    colors = []
    for idx in filtered_metrics.index:
        if highlight_stock and idx == highlight_stock:
            colors.append('red')  # Highlighted stock is always red
        elif similarity_map is not None and similarity_threshold is not None:
            # Check similarity score for yellow highlighting
            idx_normalized = ai_comps_normalize_symbol(idx)
            idx_short = idx.replace('.NS', '').upper()
            score = similarity_map.get(idx_normalized) or similarity_map.get(idx_short) or similarity_map.get(idx)
            if score is not None and score >= similarity_threshold:
                colors.append('yellow')  # Similar stocks are yellow
            else:
                colors.append('steelblue')  # Default color
        else:
            colors.append('steelblue')  # Default color when no similarity data
    
    sizes = [base_sizes.loc[idx] for idx in filtered_metrics.index]  # No size multiplier for highlighted stock
    trace_cls = go.Scatter  # Force regular Scatter (avoid WebGL blank issue)
    fig.add_trace(trace_cls(
        x=list(filtered_metrics[x_col]*x_multiplier),
        y=list(filtered_metrics[y_col]*y_multiplier),
        mode='markers+text',
        text=[i.replace('.NS','') for i in filtered_metrics.index],
        textposition='top center',
        textfont=dict(size=8),
        marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(color='black', width=0.6)),
        hovertemplate='%{hovertext}<extra></extra>', hovertext=hover_text, name='Stocks'
    ))
    # If somehow sizes produced no visible markers, add fallback constant-size layer
    if len(fig.data[0]['marker']['size']) == 0 or all((s is None or (isinstance(s,float) and s <= 0)) for s in fig.data[0]['marker']['size']):
        fig.add_trace(go.Scatter(
            x=list(filtered_metrics[x_col]*x_multiplier),
            y=list(filtered_metrics[y_col]*y_multiplier),
            mode='markers',
            marker=dict(size=14, color='orange'),
            hovertemplate='%{hovertext}<extra></extra>', hovertext=hover_text, name='Fallback'
        ))
    # Medians and quadrants (unchanged)
    fig.add_hline(y=median_y*y_multiplier, line_dash='dash', line_color='red', annotation_text=f"Median {y_title}: {median_y*y_multiplier:.2f}", annotation_position='top right')
    fig.add_vline(x=median_x*x_multiplier, line_dash='dash', line_color='red', annotation_text=f"Median {x_title}: {median_x*x_multiplier:.2f}", annotation_position='top left')
    fig.add_shape(type='rect', layer='below', x0=median_x*x_multiplier, y0=median_y*y_multiplier, x1=filtered_metrics[x_col].max()*x_multiplier, y1=filtered_metrics[y_col].max()*y_multiplier, fillcolor='lightgreen', opacity=0.08, line_width=0)
    fig.add_shape(type='rect', layer='below', x0=filtered_metrics[x_col].min()*x_multiplier, y0=median_y*y_multiplier, x1=median_x*x_multiplier, y1=filtered_metrics[y_col].max()*y_multiplier, fillcolor='gold', opacity=0.08, line_width=0)
    fig.add_shape(type='rect', layer='below', x0=median_x*x_multiplier, y0=filtered_metrics[y_col].min()*y_multiplier, x1=filtered_metrics[x_col].max()*x_multiplier, y1=median_y*y_multiplier, fillcolor='lightcoral', opacity=0.08, line_width=0)
    fig.add_shape(type='rect', layer='below', x0=filtered_metrics[x_col].min()*x_multiplier, y0=filtered_metrics[y_col].min()*y_multiplier, x1=median_x*x_multiplier, y1=median_y*y_multiplier, fillcolor='lightgray', opacity=0.08, line_width=0)
    if autoscale:
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
    else:
        fig.update_xaxes(range=[filtered_metrics[x_col].min()*x_multiplier*0.95, filtered_metrics[x_col].max()*x_multiplier*1.05])
        fig.update_yaxes(range=[filtered_metrics[y_col].min()*y_multiplier*0.95, filtered_metrics[y_col].max()*y_multiplier*1.05])
    fig.update_layout(title=f'{industry_filter}: {x_title} vs {y_title}<br><sub>Total Stocks: {len(filtered_metrics)}</sub>', xaxis_title=x_title, yaxis_title=y_title, showlegend=False, width=800, height=600, hovermode='closest')
    if debug:
        fig.add_annotation(xref='paper', yref='paper', x=0, y=1.18, showarrow=False, text=f"DEBUG pts={len(filtered_metrics)}")
    if len(fig.data)==0:
        fig.add_trace(go.Scatter(x=[median_x*x_multiplier], y=[median_y*y_multiplier], mode='markers', marker=dict(color='red', size=20)))
    return fig, filtered_metrics


def render_charts(df, filtered_df, chart_configs, remove_outliers, highlight_stock, title_prefix, debug_mode=False, use_gl=True, similarity_threshold=None, similarity_map=None, ai_plot_mode="Highlight Mode"):
    """
    Render scatter charts using the existing plotting pipeline.
    This is a helper function to avoid code duplication across modes.
    ai_plot_mode: "Highlight Mode" - all stocks plotted, similar ones yellow
                  "Filter Mode" - only similar stocks plotted
    """
    num_charts = len(chart_configs)
    
    if filtered_df is None or len(filtered_df) == 0:
        st.warning(f"No stocks found for {title_prefix}.")
        return
    
    # Apply Filter Mode if enabled - filter to only stocks meeting similarity threshold
    df_to_plot = filtered_df
    if ai_plot_mode == "Filter Mode" and similarity_map and similarity_threshold is not None:
        # Filter to only include stocks that meet the similarity threshold
        filtered_indices = []
        for idx in filtered_df.index:
            idx_normalized = ai_comps_normalize_symbol(idx)
            idx_short = idx.replace('.NS', '').upper()
            score = similarity_map.get(idx_normalized) or similarity_map.get(idx_short) or similarity_map.get(idx)
            # Include if score >= threshold OR if it's the highlighted stock
            if (score is not None and score >= similarity_threshold) or (highlight_stock and idx == highlight_stock):
                filtered_indices.append(idx)
        
        if filtered_indices:
            df_to_plot = filtered_df.loc[filtered_indices]
            st.info(f"🎯 **Filter Mode**: Showing {len(df_to_plot)} stocks with similarity ≥ {similarity_threshold}%")
        else:
            st.warning(f"No stocks meet the similarity threshold of {similarity_threshold}%. Try lowering the threshold.")
            return
    
    # Layout based on number of charts
    if num_charts == 1:
        cols = [st.container()]
    elif num_charts == 2:
        cols = st.columns(2)
    elif num_charts == 3:
        cols = st.columns([1, 1, 1])
    else:
        col1, col2 = st.columns(2)
        cols = [col1, col2, col1, col2]
    
    all_final_metrics = []
    
    for i, cfg in enumerate(chart_configs):
        x_col = cfg['x_col']
        y_col = cfg['y_col']
        chart_num = cfg['chart_num']
        
        metrics_df, message = extract_metrics(df_to_plot, x_col, y_col)
        
        if metrics_df is None:
            with cols[i % len(cols)]:
                st.error(f"Chart {chart_num} - Error extracting metrics: {message}")
            continue
        
        fig, final_metrics = create_plotly_scatter(
            metrics_df, title_prefix, x_col, y_col,
            remove_outliers, highlight_stock, autoscale=True,
            use_gl=use_gl, debug=debug_mode,
            similarity_threshold=similarity_threshold, similarity_map=similarity_map
        )
        
        if fig is None:
            with cols[i % len(cols)]:
                st.warning(f"Chart {chart_num} - No data available after filtering.")
            continue
        
        with cols[i % len(cols)]:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📊 Chart {chart_num}: {message}")
            
            if highlight_stock and highlight_stock in final_metrics.index:
                sd = final_metrics.loc[highlight_stock]
                xv = sd[x_col] if pd.notna(sd[x_col]) else "N/A"
                yv = sd[y_col] if pd.notna(sd[y_col]) else "N/A"
                st.info(f"🎯 **{highlight_stock.replace('.NS', '')}**: {x_col}={xv}, {y_col}={yv}")
            
            all_final_metrics.append(final_metrics)
    
    # Download options
    if all_final_metrics:
        st.markdown("---")
        st.subheader("📥 Download Options")
        download_cols = st.columns(min(len(all_final_metrics), 4))
        
        for i, (fm, cfg) in enumerate(zip(all_final_metrics, chart_configs)):
            with download_cols[i % len(download_cols)]:
                dl = fm.copy()
                dl.index.name = 'Stock_Symbol'
                dl = dl.reset_index()
                csv = dl.to_csv(index=False)
                safe_title = title_prefix.replace(' ', '_').replace('/', '_')
                st.download_button(
                    label=f"📥 Chart {cfg['chart_num']} Data",
                    data=csv,
                    file_name=f"{safe_title}_{cfg['x_col']}_vs_{cfg['y_col']}_analysis.csv",
                    mime="text/csv",
                    key=f"download_{i}"
                )
        
        st.info("💡 **Tip**: Hover over points in the charts for detailed information!")


def main():
    """Main Streamlit app with URL parameter handling and Mode Selector"""
    
    # Handle URL parameters first
    navigation_info = handle_url_parameters()
    url_mode = navigation_info.get('mode')  # 'scatter', 'concall', or None
    
    # Load data
    with st.spinner("Loading stock data..."):
        df, error = load_and_process_data()
    
    if df is None:
        st.error(f"Error loading data: {error}")
        st.info("Make sure 'stocks_info.csv' exists in the current directory.")
        return
    
    # Load index constituents
    index_dict, index_error = load_index_constituents()
    
    # Show data info
    num_stocks = len(df)
    num_metrics = len([col for col in df.columns if col not in ['industry', 'longName', 'shortName']])
    
    # Handle navigation from URL parameters
    nav_mode, nav_stock, nav_industry = display_navigation_banner(navigation_info, df)
    
    # Get available numeric columns
    numeric_columns = get_numeric_columns(df)
    
    if not numeric_columns:
        st.error("No numeric columns found in the dataset.")
        return
    
    # Get available industries
    industries = get_available_industries(df)
    
    if not industries:
        st.error("No industry data found in the dataset.")
        return
    
    # ==================== MODE SELECTOR ====================
    st.sidebar.header("🔄 Analysis Mode")
    
    analysis_mode = st.sidebar.radio(
        "Select Mode:",
        ["Stock Mode", "Sector Mode", "Index Mode"],
        index=0,
        help="Choose how to select stocks for analysis"
    )
    
    st.sidebar.markdown("---")
    
    # Variables to hold selection results
    selected_stock = None
    selected_industry = None
    selected_index = None
    filtered_df = None
    title_prefix = ""
    highlight_stock = None
    
    # ==================== STOCK MODE ====================
    if analysis_mode == "Stock Mode":
        st.sidebar.header("📈 Stock Selection")
        
        # Get available stocks
        available_stocks = sorted([stock.replace('.NS', '') for stock in df.index if pd.notna(df.loc[stock, 'industry'])])
        
        if not available_stocks:
            st.error("No stocks with industry data found.")
            return
        
        # Use URL parameter if available
        if nav_stock:
            nav_stock_clean = nav_stock.replace('.NS', '')
            try:
                stock_index = available_stocks.index(nav_stock_clean)
            except ValueError:
                stock_index = 0
        else:
            stock_index = 0
        
        selected_stock_clean = st.sidebar.selectbox(
            "📈 Select Stock:",
            available_stocks,
            index=stock_index,
            help="Choose a stock to analyze its industry"
        )
        
        if selected_stock_clean:
            selected_stock = f"{selected_stock_clean}.NS"
            
            if selected_stock in df.index:
                selected_industry = df.loc[selected_stock, 'industry']
                st.sidebar.success(f"🎯 **{selected_stock_clean}** belongs to **{selected_industry}** industry")
                
                # Filter by the stock's industry
                filtered_df = filter_by_industry(df, selected_industry)
                title_prefix = f"{selected_industry} Industry"
                highlight_stock = selected_stock
            else:
                st.sidebar.error(f"Stock {selected_stock_clean} not found in data")
                return
        
        # ==================== AI COMPS CONFIG (Stock Mode Only) ====================
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Comps Config")
        
        if not GENAI_AVAILABLE:
            st.sidebar.error("❌ google-generativeai not installed")
        else:
            # Show current API keys status
            num_keys = len(st.session_state.ai_comps_api_keys)
            if num_keys > 0:
                st.sidebar.success(f"✅ {num_keys} API key(s) configured")
                current_key_idx = st.session_state.ai_comps_current_key_index % num_keys
                st.sidebar.caption(f"Next key to use: Key {current_key_idx + 1}")
            else:
                st.sidebar.warning("⚠️ No API keys configured")
                st.sidebar.caption("Add keys to DEFAULT_API_KEYS in code")
            
            # Option to add more keys at runtime
            with st.sidebar.expander("➕ Add API Key"):
                new_key = st.text_input("Gemini API Key:", type="password", key="sidebar_new_api_key", placeholder="Paste key here...")
                if st.button("Add Key", key="sidebar_add_key_btn"):
                    if new_key and new_key not in st.session_state.ai_comps_api_keys:
                        st.session_state.ai_comps_api_keys.append(new_key)
                        st.sidebar.success("✅ Key added!")
                        st.rerun()
                    elif new_key in st.session_state.ai_comps_api_keys:
                        st.sidebar.warning("Key already exists")
            
            # Batch size config
            ai_batch_size = st.sidebar.slider(
                "Batch size:",
                min_value=1,
                max_value=10,
                value=5,
                help="Companies per API call",
                key="sidebar_ai_batch"
            )
            
            # Run Analysis button
            if st.sidebar.button("🚀 Run AI Analysis", type="primary", key="sidebar_run_ai"):
                if filtered_df is not None and len(filtered_df) > 0:
                    if st.session_state.ai_comps_api_keys:
                        target_stock_name = selected_stock_clean
                        target_bd = df.loc[selected_stock, 'longName'] if selected_stock in df.index else title_prefix
                        
                        with st.spinner("Running AI analysis..."):
                            results = ai_comps_run_analysis(
                                filtered_df, 
                                target_bd, 
                                target_stock_name, 
                                ai_batch_size
                            )
                            if results is not None:
                                st.session_state.ai_comps_results_df = results
                                st.rerun()
                    else:
                        st.sidebar.error("Add API keys first!")
                else:
                    st.sidebar.error("Select a stock first!")
    
    # ==================== SECTOR MODE ====================
    elif analysis_mode == "Sector Mode":
        st.sidebar.header("🏭 Sector Selection")
        
        # Use URL parameter if available
        if nav_industry:
            try:
                industry_index = industries.index(nav_industry)
            except ValueError:
                industry_index = 0
        else:
            industry_index = 0
        
        selected_industry = st.sidebar.selectbox(
            "🏭 Select Sector/Industry:",
            industries,
            index=industry_index,
            help="Choose a sector to view all stocks in that sector"
        )
        
        if selected_industry:
            filtered_df = filter_by_industry(df, selected_industry)
            title_prefix = f"{selected_industry} Sector"
            highlight_stock = None  # No specific stock highlighted in sector mode
            
            if filtered_df is not None:
                st.sidebar.success(f"📊 **{len(filtered_df)}** stocks in **{selected_industry}**")
    
    # ==================== INDEX MODE ====================
    elif analysis_mode == "Index Mode":
        st.sidebar.header("📑 Index Selection")
        
        if index_dict is None:
            st.sidebar.error(f"Could not load index data: {index_error}")
            st.error("Index constituents file not found. Please ensure 'index_constituents_long.csv' exists.")
            return
        
        # Get available indices
        available_indices = sorted(index_dict.keys())
        
        if not available_indices:
            st.sidebar.error("No indices found in the constituents file.")
            return
        
        selected_index = st.sidebar.selectbox(
            "📑 Select Index:",
            available_indices,
            index=0,
            help="Choose an index to view all constituent stocks"
        )
        
        if selected_index:
            index_symbols = index_dict[selected_index]
            
            # Filter df to only include stocks in the index
            filtered_df = filter_by_symbols(df, index_symbols)
            title_prefix = f"{selected_index} Index"
            highlight_stock = None  # No specific stock highlighted in index mode
            
            # Show stats
            matched_count = len(filtered_df)
            total_in_index = len(index_symbols)
            st.sidebar.success(f"📊 **{matched_count}** of **{total_in_index}** stocks found in data")
            
            if matched_count < total_in_index:
                missing_count = total_in_index - matched_count
                st.sidebar.warning(f"⚠️ {missing_count} stocks not in stocks_info.csv")
    
    # ==================== CHART CONFIGURATION ====================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Chart Configuration")
    
    debug_mode = False
    use_gl = True
    
    # Only build chart configuration if charts will be shown
    if url_mode != 'concall':
        num_charts = st.sidebar.slider(
            "📈 Number of Charts:",
            min_value=1,
            max_value=5,
            value=4,
            help="Choose how many charts to display"
        )
        
        # Store chart configurations
        chart_configs = []
        
        for i in range(num_charts):
            st.sidebar.markdown(f"#### Chart {i+1}")
            
            # Default values for each chart
            defaults_x = ['trailingPE', 'trailingPE', 'profitMargins', 'debtToEquity']
            defaults_y = ['earningsGrowth', 'revenueGrowth', 'operatingMargins', 'profitMargins']

            default_x = defaults_x[i] if i < len(defaults_x) and defaults_x[i] in numeric_columns else numeric_columns[0]
            default_y = defaults_y[i] if i < len(defaults_y) and defaults_y[i] in numeric_columns else (
                numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
            )
            
            x_col = st.sidebar.selectbox(
                f"🔶 X-Axis (Chart {i+1}):",
                numeric_columns,
                index=numeric_columns.index(default_x) if default_x in numeric_columns else 0,
                help=f"Choose the X-axis metric for Chart {i+1}",
                key=f"x_axis_{i}"
            )
            
            y_col = st.sidebar.selectbox(
                f"🔶 Y-Axis (Chart {i+1}):",
                numeric_columns,
                index=numeric_columns.index(default_y) if default_y in numeric_columns else 0,
                help=f"Choose the Y-axis metric for Chart {i+1}",
                key=f"y_axis_{i}"
            )
            
            chart_configs.append({
                'x_col': x_col,
                'y_col': y_col,
                'chart_num': i + 1
            })
        
        # Additional filters
        remove_outliers = st.sidebar.checkbox(
            "🎯 Remove Outliers",
            value=True,
            help="Remove extreme values for better visualization"
        )
        
        # ==================== AI COMPS SIMILARITY CONTROLS ====================
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Similarity Analysis")
        
        # Plot mode selector
        ai_plot_mode = st.sidebar.radio(
            "Plot Mode:",
            ["Highlight Mode", "Filter Mode"],
            index=0 if st.session_state.ai_comps_plot_mode == "Highlight Mode" else 1,
            help="Highlight Mode: All stocks plotted, similar ones in yellow. Filter Mode: Only similar stocks plotted.",
            key="ai_plot_mode_radio"
        )
        st.session_state.ai_comps_plot_mode = ai_plot_mode
        
        # Similarity threshold slider
        similarity_threshold = st.sidebar.slider(
            "Similarity threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.ai_comps_similarity_threshold,
            help="Stocks with relevance >= this value will be highlighted (Highlight Mode) or shown (Filter Mode)."
        )
        st.session_state.ai_comps_similarity_threshold = similarity_threshold
        
        # Get similarity map if available
        similarity_map = ai_comps_get_similarity_map()
        
        # Show status
        if similarity_map:
            num_similar = sum(1 for s, score in similarity_map.items() if score >= similarity_threshold and s.endswith('.NS'))
            st.sidebar.success(f"✅ AI data: {len(similarity_map)//2} stocks")
            if ai_plot_mode == "Filter Mode":
                st.sidebar.info(f"🎯 {num_similar} stocks ≥ {similarity_threshold}% threshold")
        else:
            st.sidebar.info("💡 Run AI comps to enable similarity highlighting")
    else:
        num_charts = 0
        chart_configs = []
        remove_outliers = True
        similarity_threshold = None
        similarity_map = None
    
    # # ==================== STATUS DISPLAY ====================
    # if filtered_df is not None and len(filtered_df) > 0:
    #     status_text = f"📊 **{analysis_mode}**: Plotting **{len(filtered_df)}** stocks"
    #     if analysis_mode == "Stock Mode" and selected_industry:
    #         status_text += f" from **{selected_industry}** industry"
    #     elif analysis_mode == "Sector Mode" and selected_industry:
    #         status_text += f" in **{selected_industry}** sector"
    #     elif analysis_mode == "Index Mode" and selected_index:
    #         status_text += f" from **{selected_index}** index"
        
    #     st.markdown(status_text)
    #     st.markdown("---")
    
    # ==================== RENDERING LOGIC ====================
    if url_mode == 'scatter':
        # Direct scatter mode from URL
        if filtered_df is not None and len(filtered_df) > 0:
            render_charts(df, filtered_df, chart_configs, remove_outliers, highlight_stock, title_prefix, debug_mode, use_gl, similarity_threshold, similarity_map, st.session_state.ai_comps_plot_mode)
        else:
            st.warning("Please select a stock, sector, or index to view charts.")
    
    elif url_mode == 'concall':
        # Direct concall mode from URL
        st.subheader("🗣️ Concall / Summary Viewer")
        summaries_df, summaries_error = load_concall_summaries()
        
        if summaries_df is None:
            st.warning(summaries_error)
        else:
            symbol_col = next((c for c in ["Stock", "Symbol", "Ticker"] if c in summaries_df.columns), None)
            summary_col = next((c for c in ["Summary", "Concall", "ConcallSummary", "Notes"] if c in summaries_df.columns), None)
            stock_symbols = sorted(summaries_df[symbol_col].unique())
            display_symbols = [s.replace('.NS', '') for s in stock_symbols]
            
            default_idx = 0
            if 'type' in navigation_info and navigation_info['type'] == 'stock' and navigation_info['value'] in stock_symbols:
                default_idx = stock_symbols.index(navigation_info['value'])
            
            chosen_display = st.selectbox("Select Stock:", display_symbols, index=default_idx)
            chosen_symbol = chosen_display + '.NS'
            row = summaries_df[summaries_df[symbol_col] == chosen_symbol]
            
            if row.empty:
                st.info("No summary available for this stock.")
            else:
                r = row.iloc[0]
                st.markdown(f"### {chosen_display}")
                
                if summary_col and pd.notna(r.get(summary_col)):
                    st.markdown(r.get(summary_col))
                else:
                    for c in summaries_df.columns:
                        if c != symbol_col and pd.notna(r.get(c)):
                            st.markdown(f"**{c}**: {r.get(c)}")
                
                with st.expander("🔍 Search across summaries"):
                    q = st.text_input("Keyword")
                    if q:
                        hits = summaries_df[summaries_df.apply(
                            lambda rw: any(q.lower() in str(rw[col]).lower() for col in summaries_df.columns if col != symbol_col),
                            axis=1
                        )]
                        if hits.empty:
                            st.info("No matches.")
                        else:
                            cols_to_show = [symbol_col] + ([summary_col] if summary_col else [])
                            st.dataframe(hits[cols_to_show])
    
    else:
        # Default: show both tabs
        charts_tab, concall_tab, ai_comps_tab = st.tabs(["📊 Charts", "🗣️ Concall Summaries", "🤖 AI Comps"])
        
        with charts_tab:
            if filtered_df is not None and len(filtered_df) > 0:
                render_charts(df, filtered_df, chart_configs, remove_outliers, highlight_stock, title_prefix, debug_mode, use_gl, similarity_threshold, similarity_map, st.session_state.ai_comps_plot_mode)
            else:
                st.warning("Please select a stock, sector, or index to view charts.")
        
        with concall_tab:
            st.subheader("🗣️ Concall / Summary Viewer")
            summaries_df, summaries_error = load_concall_summaries()
            
            if summaries_df is None:
                st.warning(summaries_error)
            else:
                symbol_col = next((c for c in ["Stock", "Symbol", "Ticker"] if c in summaries_df.columns), None)
                summary_col = next((c for c in ["Summary", "Concall", "ConcallSummary", "Notes"] if c in summaries_df.columns), None)
                stock_symbols = sorted(summaries_df[symbol_col].unique())
                display_symbols = [s.replace('.NS', '') for s in stock_symbols]
                
                default_idx = 0
                if 'type' in navigation_info and navigation_info['type'] == 'stock' and navigation_info['value'] in stock_symbols:
                    default_idx = stock_symbols.index(navigation_info['value'])
                
                chosen_display = st.selectbox("Select Stock:", display_symbols, index=default_idx, key="concall_stock_selector")
                chosen_symbol = chosen_display + '.NS'
                row = summaries_df[summaries_df[symbol_col] == chosen_symbol]
                
                if row.empty:
                    st.info("No summary available for this stock.")
                else:
                    r = row.iloc[0]
                    st.markdown(f"### {chosen_display}")
                    
                    if summary_col and pd.notna(r.get(summary_col)):
                        st.markdown(r.get(summary_col))
                    else:
                        for c in summaries_df.columns:
                            if c != symbol_col and pd.notna(r.get(c)):
                                st.markdown(f"**{c}**: {r.get(c)}")
                    
                    with st.expander("🔍 Search across summaries"):
                        q = st.text_input("Keyword", key="concall_search")
                        if q:
                            hits = summaries_df[summaries_df.apply(
                                lambda rw: any(q.lower() in str(rw[col]).lower() for col in summaries_df.columns if col != symbol_col),
                                axis=1
                            )]
                            if hits.empty:
                                st.info("No matches.")
                            else:
                                cols_to_show = [symbol_col] + ([summary_col] if summary_col else [])
                                st.dataframe(hits[cols_to_show])
        
        with ai_comps_tab:
            st.subheader("🤖 AI Competitor Analysis Results")
            
            # Check mode - AI Comps only available in Stock Mode
            if analysis_mode != "Stock Mode":
                st.info("🔒 AI Competitor Analysis is only available in **Stock Mode**. Please switch to Stock Mode to use this feature.")
            elif not GENAI_AVAILABLE:
                st.error("❌ google-generativeai package not installed. Run: `pip install google-generativeai`")
            elif st.session_state.ai_comps_results_df is None:
                st.info("� No AI analysis results yet. Use the **AI Comps Config** section in the sidebar to run analysis.")
                st.markdown("""
                **How to use AI Comps:**
                1. Select a stock in Stock Mode
                2. Configure API keys in the sidebar (or add to `DEFAULT_API_KEYS` in code)
                3. Click **Run AI Analysis** in the sidebar
                4. Results will appear here with similarity highlighting on charts
                """)
            else:
                # Display results
                results_df = st.session_state.ai_comps_results_df
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Analyzed", len(results_df))
                col2.metric("High Similarity (≥70)", len(results_df[results_df['Relevance Score'] >= 70]))
                col3.metric("Medium (50-69)", len(results_df[(results_df['Relevance Score'] >= 50) & (results_df['Relevance Score'] < 70)]))
                col4.metric("Avg Score", f"{results_df['Relevance Score'].mean():.1f}")
                
                st.markdown("---")
                
                # Results table
                st.dataframe(
                    results_df[['Company Name', 'Relevance Score', 'Business Summary', 'Business Model', 'Relevance Reason']],
                    use_container_width=True,
                    height=400
                )
                
                # Action buttons
                col_dl, col_clear = st.columns(2)
                
                with col_dl:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_data,
                        f"ai_comps_results_{title_prefix.replace(' ', '_')}.csv",
                        "text/csv",
                        key="ai_comps_download"
                    )
                
                with col_clear:
                    if st.button("🗑️ Clear Results", key="ai_comps_clear"):
                        st.session_state.ai_comps_results_df = None
                        st.rerun()


if __name__ == "__main__":
    main()

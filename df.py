import streamlit as st
import yfinance as yf
import pandas as pd
from nselib import capital_market
import time
import requests
import base64

# Config
DELAY = 0.1
MAX_RETRIES = 3
BATCH_SIZE = 100
OUTPUT_FILE = "stocks_info.csv"

# ====== DEFAULT GITHUB CONFIG (Edit these) ======
DEFAULT_GITHUB_TOKEN = "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"  # Your token here: ghp_xxxxx
DEFAULT_GITHUB_REPO = "kanishkchawla2/ScatterPlot"   # Your repo: username/repo-name
DEFAULT_GITHUB_BRANCH = "main"
# =================================================

st.set_page_config(page_title="NSE Stock Data Fetcher", layout="wide")
st.title("📈 NSE Stock Data Fetcher")

# GitHub Setup
st.sidebar.header("⚙️ GitHub Setup")

use_default = st.sidebar.checkbox("Use default GitHub config", value=True if DEFAULT_GITHUB_TOKEN else False)

if use_default and DEFAULT_GITHUB_TOKEN:
    github_token = DEFAULT_GITHUB_TOKEN
    github_repo = DEFAULT_GITHUB_REPO
    github_branch = DEFAULT_GITHUB_BRANCH
    st.sidebar.success(f"Using default: {DEFAULT_GITHUB_REPO}")
else:
    st.sidebar.markdown("""
**To save files to your GitHub repo:**
1. Go to [GitHub Tokens](https://github.com/settings/tokens)
2. Generate token with **repo** scope
3. Enter details below
""")
    github_token = st.sidebar.text_input("GitHub Token", type="password")
    github_repo = st.sidebar.text_input("Repository", placeholder="username/repo-name")
    github_branch = st.sidebar.text_input("Branch", value="main")

def verify_github_config(token, repo, branch):
    """Verify GitHub token and repo access before fetching data"""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check token validity
    user_response = requests.get("https://api.github.com/user", headers=headers)
    if user_response.status_code != 200:
        return False, "❌ Invalid GitHub token"
    
    # Check repo access
    repo_response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
    if repo_response.status_code == 404:
        return False, f"❌ Repository '{repo}' not found"
    elif repo_response.status_code != 200:
        return False, f"❌ Cannot access repository '{repo}'"
    
    # Check write permission by checking if we can get repo contents
    contents_response = requests.get(f"https://api.github.com/repos/{repo}/contents", headers=headers)
    if contents_response.status_code not in [200, 404]:  # 404 is ok for empty repo
        return False, "❌ No write permission to repository"
    
    # Check branch exists
    branch_response = requests.get(f"https://api.github.com/repos/{repo}/branches/{branch}", headers=headers)
    if branch_response.status_code == 404:
        return False, f"❌ Branch '{branch}' not found"
    
    return True, f"✅ Connected to {repo} ({branch})"

def save_to_github(content, filename, token, repo, branch):
    """Save file to GitHub repo using API"""
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists (to get SHA for update)
    response = requests.get(url, headers=headers)
    sha = response.json().get("sha") if response.status_code == 200 else None
    
    # Prepare content
    encoded_content = base64.b64encode(content.encode()).decode()
    
    data = {
        "message": f"Update {filename} - {time.strftime('%Y-%m-%d %H:%M')}",
        "content": encoded_content,
        "branch": branch
    }
    if sha:
        data["sha"] = sha
    
    # Push to GitHub
    response = requests.put(url, headers=headers, json=data)
    if response.status_code in [200, 201]:
        return True, "✅ Saved successfully"
    else:
        error_msg = response.json().get("message", "Unknown error")
        return False, f"❌ GitHub API error: {error_msg}"

def get_nse_symbols():
    """Get list of NSE equity symbols"""
    try:
        df = capital_market.equity_list()
        symbols = df['SYMBOL'].tolist()
        return [f"{s}.NS" for s in symbols]
    except Exception as e:
        st.error(f"Error fetching NSE symbols: {e}")
        return []

def get_stock_info(symbol):
    """Fetch stock info from yfinance"""
    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'symbol' in info:
                return info
        except Exception:
            time.sleep(DELAY * (attempt + 1))
    return None

def run_fetcher():
    """Main fetcher logic"""
    # Validate GitHub config
    if not github_token or not github_repo:
        st.error("⚠️ Please enter GitHub Token and Repository in the sidebar!")
        return
    
    # Verify GitHub connection FIRST
    status = st.empty()
    status.info("🔐 Verifying GitHub connection...")
    
    valid, message = verify_github_config(github_token, github_repo, github_branch)
    if not valid:
        st.error(message)
        return
    
    st.success(message)
    
    # Now fetch data
    status.info("Fetching NSE symbols...")
    symbols = get_nse_symbols()
    
    if not symbols:
        st.error("No symbols found!")
        return
    
    st.success(f"Found {len(symbols)} symbols")
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    all_data = []
    for i, symbol in enumerate(symbols):
        progress = (i + 1) / len(symbols)
        progress_bar.progress(progress)
        progress_text.text(f"Processing {symbol} ({i+1}/{len(symbols)})")
        
        info = get_stock_info(symbol)
        if info:
            all_data.append(info)
        
        time.sleep(DELAY)
    
    if all_data:
        df = pd.DataFrame(all_data)
        csv_content = df.to_csv(index=False)
        
        progress_text.text("Saving to GitHub...")
        success, message = save_to_github(csv_content, OUTPUT_FILE, github_token, github_repo, github_branch)
        if success:
            progress_bar.progress(1.0)
            progress_text.text("Done!")
            st.success(f"✅ Saved {len(all_data)} stocks to GitHub: {github_repo}/{OUTPUT_FILE}")
            st.dataframe(df.head(20))
        else:
            st.error(message)
    else:
        st.error("No data fetched!")

# Simple button
if st.button("🚀 Fetch Stock Data", type="primary", use_container_width=True):
    run_fetcher()

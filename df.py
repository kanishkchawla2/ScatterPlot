import streamlit as st
import yfinance as yf
import pandas as pd
import os
import time
import logging
import base64
import requests
from datetime import datetime

# Page config
st.set_page_config(page_title="NSE Stock Data Fetcher", page_icon="📈", layout="wide")

# Disable yfinance debug logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)

st.title("📈 NSE Stock Data Fetcher")
st.markdown("Fetches stock info from Yahoo Finance and saves to GitHub")

# GitHub Configuration in sidebar
st.sidebar.header("🔧 GitHub Configuration")
github_token = st.sidebar.text_input("GitHub Token", type="password", help="Personal Access Token with repo scope")
github_repo = st.sidebar.text_input("Repository", value="username/repo", help="Format: username/repo")
github_branch = st.sidebar.text_input("Branch", value="main")
github_path = st.sidebar.text_input("File Path", value="data/stocks_info.csv", help="Path in repo to save CSV")

# Processing Configuration
st.sidebar.header("⚙️ Processing Config")
delay_between_requests = st.sidebar.slider("Delay between requests (s)", 0.0, 2.0, 0.1, 0.1)
batch_size = st.sidebar.number_input("Backup every N stocks", 50, 500, 100)
max_retries = st.sidebar.number_input("Max retries", 1, 5, 3)


def save_to_github(content: str, filename: str, token: str, repo: str, branch: str, message: str = None):
    """Save content to GitHub repository"""
    if not all([token, repo, filename]):
        return False, "Missing GitHub configuration"
    
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists (to get SHA for update)
    sha = None
    try:
        response = requests.get(url, headers=headers, params={"ref": branch})
        if response.status_code == 200:
            sha = response.json().get("sha")
    except:
        pass
    
    # Prepare content
    content_encoded = base64.b64encode(content.encode()).decode()
    
    if message is None:
        message = f"Update {filename} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    data = {
        "message": message,
        "content": content_encoded,
        "branch": branch
    }
    
    if sha:
        data["sha"] = sha
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            return True, f"Successfully saved to {repo}/{filename}"
        else:
            return False, f"GitHub API error: {response.status_code} - {response.text[:200]}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def fetch_nse_symbols():
    """Fetch NSE equity symbols"""
    try:
        from nselib import capital_market
        equity_list = capital_market.equity_list()
        if 'SYMBOL' in equity_list.columns:
            symbols = [f"{symbol}.NS" for symbol in equity_list['SYMBOL']]
            return symbols, None
        else:
            return None, "SYMBOL column not found"
    except Exception as e:
        return None, str(e)


def process_stocks(symbols, delay, max_retries, progress_bar, status_text, batch_size, 
                   github_token, github_repo, github_branch, github_path):
    """Process stocks and fetch info"""
    all_stock_data = []
    successful_count = 0
    failed_count = 0
    failed_symbols = []
    
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {symbol} ({i+1}/{total}) | ✅ {successful_count} | ❌ {failed_count}")
        
        try:
            ticker = yf.Ticker(symbol)
            info_data = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    info_data = ticker.info
                    if info_data and 'symbol' in info_data:
                        break
                except Exception:
                    if attempt < max_retries:
                        time.sleep(0.3)
            
            if info_data and 'symbol' in info_data:
                df = pd.DataFrame(list(info_data.items()), columns=['Field', 'Value'])
                df['Stock'] = symbol
                df = df[['Stock', 'Field', 'Value']]
                all_stock_data.append(df)
                successful_count += 1
            else:
                failed_count += 1
                failed_symbols.append(symbol)
                
        except Exception as e:
            failed_count += 1
            failed_symbols.append(symbol)
        
        # Backup to GitHub every batch_size stocks
        if (i + 1) % batch_size == 0 and all_stock_data:
            if github_token and github_repo:
                temp_df = pd.concat(all_stock_data, ignore_index=True)
                csv_content = temp_df.to_csv(index=False)
                backup_path = github_path.replace('.csv', f'_backup_{i+1}.csv')
                save_to_github(csv_content, backup_path, github_token, github_repo, github_branch,
                              f"Backup at {i+1} stocks")
                status_text.text(f"💾 Backup saved ({i+1} processed)")
        
        if delay > 0:
            time.sleep(delay)
    
    return all_stock_data, successful_count, failed_count, failed_symbols


# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Stock Symbols")
    
    # Fetch symbols
    if st.button("🔄 Fetch NSE Symbols", use_container_width=True):
        with st.spinner("Fetching NSE equity list..."):
            symbols, error = fetch_nse_symbols()
            if symbols:
                st.session_state['symbols'] = symbols
                st.success(f"✅ Fetched {len(symbols)} symbols")
            else:
                st.error(f"❌ Error: {error}")
                st.info("You can manually enter symbols below")
    
    # Manual input option
    manual_symbols = st.text_area(
        "Or enter symbols manually (comma-separated)",
        placeholder="RELIANCE.NS, TCS.NS, INFY.NS",
        height=100
    )
    
    if manual_symbols:
        st.session_state['symbols'] = [s.strip() for s in manual_symbols.split(',') if s.strip()]

with col2:
    st.subheader("📈 Status")
    if 'symbols' in st.session_state:
        st.metric("Symbols Loaded", len(st.session_state['symbols']))
    else:
        st.metric("Symbols Loaded", 0)

st.divider()

# Process button
if st.button("🚀 Start Processing", type="primary", use_container_width=True):
    if 'symbols' not in st.session_state or not st.session_state['symbols']:
        st.error("❌ No symbols loaded. Fetch NSE symbols or enter manually.")
    elif not github_token or not github_repo:
        st.error("❌ Please configure GitHub settings in sidebar")
    else:
        symbols = st.session_state['symbols']
        
        st.info(f"Processing {len(symbols)} stocks...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        all_data, success, failed, failed_list = process_stocks(
            symbols=symbols,
            delay=delay_between_requests,
            max_retries=max_retries,
            progress_bar=progress_bar,
            status_text=status_text,
            batch_size=batch_size,
            github_token=github_token,
            github_repo=github_repo,
            github_branch=github_branch,
            github_path=github_path
        )
        
        elapsed = time.time() - start_time
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Results
        st.success(f"🎉 Processing complete in {elapsed/60:.1f} minutes!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Success", success)
        col2.metric("❌ Failed", failed)
        col3.metric("📊 Success Rate", f"{(success/len(symbols)*100):.1f}%")
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Save to GitHub
            st.subheader("💾 Saving to GitHub...")
            csv_content = final_df.to_csv(index=False)
            
            success_save, message = save_to_github(
                content=csv_content,
                filename=github_path,
                token=github_token,
                repo=github_repo,
                branch=github_branch,
                message=f"Stock data update - {success} stocks - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            if success_save:
                st.success(f"✅ {message}")
                st.balloons()
            else:
                st.error(f"❌ {message}")
                # Offer download as fallback
                st.download_button(
                    "📥 Download CSV instead",
                    csv_content,
                    "stocks_info.csv",
                    "text/csv"
                )
            
            # Preview
            st.subheader("📋 Data Preview")
            st.dataframe(final_df.head(100), use_container_width=True)
            
            # Stats
            st.subheader("📊 Records per Stock (Top 10)")
            stock_counts = final_df['Stock'].value_counts().head(10)
            st.bar_chart(stock_counts)
        
        # Show failed symbols
        if failed_list:
            with st.expander(f"❌ Failed Symbols ({len(failed_list)})"):
                st.write(", ".join(failed_list))

# Footer
st.divider()
st.caption("💡 Tip: Create a GitHub Personal Access Token with 'repo' scope at Settings > Developer settings > Personal access tokens")

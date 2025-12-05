import streamlit as st
import yfinance as yf
import pandas as pd
from nselib import capital_market
import time

# Config
DELAY = 0.1
MAX_RETRIES = 3
BATCH_SIZE = 100
OUTPUT_FILE = "stocks_info.csv"

st.set_page_config(page_title="NSE Stock Data Fetcher", layout="wide")
st.title("📈 NSE Stock Data Fetcher")

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
    status = st.empty()
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
        
        if (i + 1) % BATCH_SIZE == 0 and all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(f"stocks_info_backup_{i+1}.csv", index=False)
        
        time.sleep(DELAY)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILE, index=False)
        progress_bar.progress(1.0)
        progress_text.text("Done!")
        st.success(f"✅ Saved {len(all_data)} stocks to {OUTPUT_FILE}")
        st.dataframe(df.head(20))
    else:
        st.error("No data fetched!")

# Simple button
if st.button("🚀 Fetch Stock Data", type="primary", use_container_width=True):
    run_fetcher()

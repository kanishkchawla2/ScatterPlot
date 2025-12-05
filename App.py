import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

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
</style>
""", unsafe_allow_html=True)

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

def create_plotly_scatter(metrics_df, industry_filter, x_col, y_col, remove_outliers=True, highlight_stock=None, autoscale=False, use_gl=True, debug=False):
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
        hover_info = f"<b>{stock_name}</b><br>Company: {company_name}<br>Market Cap: ₹{mcap_str}<br>{x_title}: {row[x_col]*x_multiplier:.2f}<br>{y_title}: {row[y_col]*y_multiplier:.2f}"; hover_text.append(hover_info)
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
    colors = ['red' if (highlight_stock and idx==highlight_stock) else 'steelblue' for idx in filtered_metrics.index]
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
    fig.update_layout(title=f'{industry_filter} Industry: {x_title} vs {y_title}<br><sub>Total Stocks: {len(filtered_metrics)}</sub>', xaxis_title=x_title, yaxis_title=y_title, showlegend=False, width=800, height=600, hovermode='closest')
    if debug:
        fig.add_annotation(xref='paper', yref='paper', x=0, y=1.18, showarrow=False, text=f"DEBUG pts={len(filtered_metrics)}")
    if len(fig.data)==0:
        fig.add_trace(go.Scatter(x=[median_x*x_multiplier], y=[median_y*y_multiplier], mode='markers', marker=dict(color='red', size=20)))
    return fig, filtered_metrics



def main():
    """Main Streamlit app with URL parameter handling"""
    
    # Handle URL parameters first
    navigation_info = handle_url_parameters()
    mode = navigation_info.get('mode')  # 'scatter', 'concall', or None
    
    # Header
    
   
    # Load data
    with st.spinner("Loading stock data..."):
        df, error = load_and_process_data()
    
    if df is None:
        st.error(f"Error loading data: {error}")
        st.info("Make sure 'stocks_info.csv' exists in the current directory.")
        return
    
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
    
    # Sidebar
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Get available industries
    industries = get_available_industries(df)
    
    if not industries:
        st.error("No industry data found in the dataset.")
        return
    
    # Always use "By Stock" mode
    selection_mode = "By Stock"
    
    # Industry selection is determined by the selected stock
    industry_index = 0
    
    # Industry selection is determined by the selected stock (hidden from user)
    # selected_industry will be set after stock selection
    
    # Stock selection - use URL parameter if available
    available_stocks = sorted([stock.replace('.NS', '') for stock in df.index if pd.notna(df.loc[stock, 'industry'])])
    
    if not available_stocks:
        st.error("No stocks with industry data found.")
        return
    
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
        # Get the industry of the selected stock
        if selected_stock in df.index:
            selected_industry = df.loc[selected_stock, 'industry']
            st.sidebar.success(f"🎯 **{selected_stock_clean}** belongs to **{selected_industry}** industry")
        else:
            st.sidebar.error(f"Stock {selected_stock_clean} not found in data")
            return
    
    # Chart configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Chart Configuration")
    debug_mode = False
    use_gl = True
    # Only build chart configuration if charts will be shown
    if mode != 'concall':
        num_charts = st.sidebar.slider(
            "📈 Number of Charts:",
            min_value=1,
            max_value=4,
            value=1,
            help="Choose how many charts to display"
        )
        # Store chart configurations
        chart_configs = []
        
        for i in range(num_charts):
            st.sidebar.markdown(f"#### Chart {i+1}")
            
            # Default values for each chart
            defaults_x = ['trailingPE', 'priceToBook', 'earningsGrowth', 'currentRatio']
            defaults_y = ['earningsGrowth', 'revenueGrowth', 'dividendYield', 'debtToEquity']
            
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
    else:
        num_charts = 0
        chart_configs = []
        remove_outliers = True
    
    # Industry info
    if selected_industry:
        industry_count = len(df[df['industry'].str.contains(selected_industry, case=False, na=False)])
        st.sidebar.info(f"📈 **{industry_count}** stocks found in **{selected_industry}** industry")
    
    # Rendering logic based on mode
    if mode == 'scatter':
        
        # ...existing chart rendering block (reuse filtered_df logic)...
        if selected_industry:
            filtered_df = filter_by_industry(df, selected_industry)
            if filtered_df is None or len(filtered_df) == 0:
                st.warning(f"No stocks found for {selected_industry} industry.")
            else:
                highlight_stock = selected_stock
                if num_charts == 1:
                    cols = [st.container()]
                elif num_charts == 2:
                    cols = st.columns(2)
                elif num_charts == 3:
                    cols = st.columns([1,1,1])
                else:
                    col1, col2 = st.columns(2); cols = [col1, col2, col1, col2]
                all_final_metrics = []
                for i, cfg in enumerate(chart_configs):
                    x_col = cfg['x_col']; y_col = cfg['y_col']; chart_num = cfg['chart_num']
                    metrics_df, message = extract_metrics(filtered_df, x_col, y_col)
                    if metrics_df is None:
                        with cols[i % len(cols)]: st.error(f"Chart {chart_num} - Error extracting metrics: {message}"); continue
                    fig, final_metrics = create_plotly_scatter(metrics_df, selected_industry, x_col, y_col, remove_outliers, highlight_stock, autoscale=True, use_gl=use_gl, debug=debug_mode)
                    if fig is None:
                        with cols[i % len(cols)]: st.warning(f"Chart {chart_num} - No data available after filtering."); continue
                    with cols[i % len(cols)]:
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"📊 Chart {chart_num}: {message}")
                        if selected_stock and selected_stock in final_metrics.index:
                            sd = final_metrics.loc[selected_stock]; xv = sd[x_col] if pd.notna(sd[x_col]) else "N/A"; yv = sd[y_col] if pd.notna(sd[y_col]) else "N/A"
                            st.info(f"🎯 **{selected_stock.replace('.NS','')}**: {x_col}={xv}, {y_col}={yv}")
                        all_final_metrics.append(final_metrics)
                if all_final_metrics:
                    st.markdown("---"); st.subheader("📥 Download Options")
                    download_cols = st.columns(min(len(all_final_metrics), 4))
                    for i, (fm, cfg) in enumerate(zip(all_final_metrics, chart_configs)):
                        with download_cols[i % len(download_cols)]:
                            dl = fm.copy(); dl.index.name = 'Stock_Symbol'; dl = dl.reset_index(); csv = dl.to_csv(index=False)
                            st.download_button(label=f"📥 Chart {cfg['chart_num']} Data", data=csv, file_name=f"{selected_industry}_{cfg['x_col']}_vs_{cfg['y_col']}_analysis.csv", mime="text/csv", key=f"download_{i}")
                    st.info("💡 **Tip**: Hover over points in the charts for detailed information!")
        else:
            st.warning("Please select an industry to view charts.")
    elif mode == 'concall':
        st.subheader("🗣️ Concall / Summary Viewer")
        summaries_df, summaries_error = load_concall_summaries()
        if summaries_df is None:
            st.warning(summaries_error)
        else:
            symbol_col = next((c for c in ["Stock","Symbol","Ticker"] if c in summaries_df.columns), None)
            summary_col = next((c for c in ["Summary","Concall","ConcallSummary","Notes"] if c in summaries_df.columns), None)
            stock_symbols = sorted(summaries_df[symbol_col].unique())
            display_symbols = [s.replace('.NS','') for s in stock_symbols]
            default_idx = 0
            if 'type' in navigation_info and navigation_info['type']=='stock' and navigation_info['value'] in stock_symbols:
                default_idx = stock_symbols.index(navigation_info['value'])
            chosen_display = st.selectbox("Select Stock:", display_symbols, index=default_idx)
            chosen_symbol = chosen_display + '.NS'
            row = summaries_df[summaries_df[symbol_col] == chosen_symbol]
            if row.empty:
                st.info("No summary available for this stock.")
            else:
                r = row.iloc[0]; st.markdown(f"### {chosen_display}")
                if summary_col and pd.notna(r.get(summary_col)):
                    st.markdown(r.get(summary_col))
                else:
                    for c in summaries_df.columns:
                        if c != symbol_col and pd.notna(r.get(c)):
                            st.markdown(f"**{c}**: {r.get(c)}")
                with st.expander("🔍 Search across summaries"):
                    q = st.text_input("Keyword")
                    if q:
                        hits = summaries_df[summaries_df.apply(lambda rw: any(q.lower() in str(rw[col]).lower() for col in summaries_df.columns if col != symbol_col), axis=1)]
                        if hits.empty:
                            st.info("No matches.")
                        else:
                            cols_to_show = [symbol_col] + ([summary_col] if summary_col else [])
                            st.dataframe(hits[cols_to_show])
    else:
        # Default: show both tabs as before
        charts_tab, concall_tab = st.tabs(["📊 Charts", "🗣️ Concall Summaries"])
        with charts_tab:
            # Reuse scatter rendering when mode not specified
            if selected_industry:
                filtered_df = filter_by_industry(df, selected_industry)
                if filtered_df is None or len(filtered_df) == 0:
                    st.warning(f"No stocks found for {selected_industry} industry.")
                else:
                    highlight_stock = selected_stock
                    if num_charts == 1:
                        cols = [st.container()]
                    elif num_charts == 2:
                        cols = st.columns(2)
                    elif num_charts == 3:
                        cols = st.columns([1,1,1])
                    else:
                        col1, col2 = st.columns(2); cols = [col1, col2, col1, col2]
                    all_final_metrics = []
                    for i, cfg in enumerate(chart_configs):
                        x_col = cfg['x_col']; y_col = cfg['y_col']; chart_num = cfg['chart_num']
                        metrics_df, message = extract_metrics(filtered_df, x_col, y_col)
                        if metrics_df is None:
                            with cols[i % len(cols)]: st.error(f"Chart {chart_num} - Error extracting metrics: {message}"); continue
                        fig, final_metrics = create_plotly_scatter(metrics_df, selected_industry, x_col, y_col, remove_outliers, highlight_stock, autoscale=True, use_gl=use_gl, debug=debug_mode)
                        if fig is None:
                            with cols[i % len(cols)]: st.warning(f"Chart {chart_num} - No data available after filtering."); continue
                        with cols[i % len(cols)]:
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"📊 Chart {chart_num}: {message}")
                            if selected_stock and selected_stock in final_metrics.index:
                                sd = final_metrics.loc[selected_stock]; xv = sd[x_col] if pd.notna(sd[x_col]) else "N/A"; yv = sd[y_col] if pd.notna(sd[y_col]) else "N/A"
                                st.info(f"🎯 **{selected_stock.replace('.NS','')}**: {x_col}={xv}, {y_col}={yv}")
                            all_final_metrics.append(final_metrics)
                    if all_final_metrics:
                        st.markdown("---"); st.subheader("📥 Download Options")
                        download_cols = st.columns(min(len(all_final_metrics), 4))
                        for i, (fm, cfg) in enumerate(zip(all_final_metrics, chart_configs)):
                            with download_cols[i % len(download_cols)]:
                                dl = fm.copy(); dl.index.name = 'Stock_Symbol'; dl = dl.reset_index(); csv = dl.to_csv(index=False)
                                st.download_button(label=f"📥 Chart {cfg['chart_num']} Data", data=csv, file_name=f"{selected_industry}_{cfg['x_col']}_vs_{cfg['y_col']}_analysis.csv", mime="text/csv", key=f"download_{i}")
                        st.info("💡 **Tip**: Hover over points in the charts for detailed information!")
            else:
                st.warning("Please select an industry to view charts.")
        with concall_tab:
            st.subheader("🗣️ Concall / Summary Viewer")
            summaries_df, summaries_error = load_concall_summaries()
            if summaries_df is None:
                st.warning(summaries_error)
            else:
                symbol_col = next((c for c in ["Stock","Symbol","Ticker"] if c in summaries_df.columns), None)
                summary_col = next((c for c in ["Summary","Concall","ConcallSummary","Notes"] if c in summaries_df.columns), None)
                stock_symbols = sorted(summaries_df[symbol_col].unique())
                display_symbols = [s.replace('.NS','') for s in stock_symbols]
                default_idx = 0
                if 'type' in navigation_info and navigation_info['type']=='stock' and navigation_info['value'] in stock_symbols:
                    default_idx = stock_symbols.index(navigation_info['value'])
                chosen_display = st.selectbox("Select Stock:", display_symbols, index=default_idx)
                chosen_symbol = chosen_display + '.NS'
                row = summaries_df[summaries_df[symbol_col] == chosen_symbol]
                if row.empty:
                    st.info("No summary available for this stock.")
                else:
                    r = row.iloc[0]; st.markdown(f"### {chosen_display}")
                    if summary_col and pd.notna(r.get(summary_col)):
                        st.markdown(r.get(summary_col))
                    else:
                        for c in summaries_df.columns:
                            if c != symbol_col and pd.notna(r.get(c)):
                                st.markdown(f"**{c}**: {r.get(c)}")
                    with st.expander("🔍 Search across summaries"):
                        q = st.text_input("Keyword")
                        if q:
                            hits = summaries_df[summaries_df.apply(lambda rw: any(q.lower() in str(rw[col]).lower() for col in summaries_df.columns if col != symbol_col), axis=1)]
                            if hits.empty:
                                st.info("No matches.")
                            else:
                                cols_to_show = [symbol_col] + ([summary_col] if summary_col else [])
                                st.dataframe(hits[cols_to_show])

if __name__ == "__main__":
    main()

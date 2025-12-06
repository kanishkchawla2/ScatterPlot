# Dynamic Stock Analysis Dashboard

An interactive Streamlit application for analyzing Indian NSE stocks, visualizing financial metrics, exploring sector-wise and index-wise performance, and viewing concall summaries. The dashboard supports multi-chart scatter plots, outlier removal, URL deep linking, and CSV downloads for analysis.

---

## Overview

This app allows users to:

* Select individual stocks and automatically analyze their industries.
* View entire sectors/industries and compare stocks using financial metrics.
* Load and visualize index constituents such as NIFTY 50.
* Generate 1–5 customizable scatter charts at a time.
* Remove outliers using the IQR method for cleaner charts.
* View uploaded concall summaries and perform keyword searches.
* Use URL parameters to deep-link directly into stocks, industries, charts, or concall mode.

---

## Features

### Analysis Modes

1. **Stock Mode**
   Select a stock. The app detects its industry, filters the data, and highlights the selected stock on all charts.

2. **Sector Mode**
   Select an industry. The app displays all stocks that belong to that sector.

3. **Index Mode**
   Select a market index. The app loads constituent symbols from an index file (CSV) and charts only those stocks.

### Scatter Chart Visualization

* Configure X and Y metrics from available numeric columns.
* Up to 5 charts can be displayed at once.
* Log-scaled marker sizes based on market cap.
* Hover tooltips show company name, market cap (in INR crore), X/Y metric values, and more.
* Median lines and quadrants for visual interpretation.

### Concall Summary Viewer

* Automatically loads summaries from `Summary.csv`, `summary.csv`, or `ConcallSummary.csv`.
* Select a stock to view its concall notes.
* Built-in keyword search across all summaries.

### URL Parameter Support

* `?stock=RELIANCE` — auto-select a stock
* `?industry=Finance` — auto-select an industry
* `?search=bank` — find stocks/industries containing the term
* `?mode=scatter` — open scatter-only mode
* `?mode=concall` — open concall viewer directly

---

## Required Data Files

Place these files in the same directory as the app.

### 1. stocks_info.csv (required)

A long-format CSV with columns:

```
Stock, Field, Value
```

Example:

```
RELIANCE,industry,Oil & Gas
RELIANCE,marketCap,2400000000000
RELIANCE,trailingPE,23.5
```

### 2. Concall Summary File (optional)

Supported filenames:

* `Summary.csv`
* `summary.csv`
* `ConcallSummary.csv`

Must contain a symbol column (`Stock`, `Symbol`, or `Ticker`).

### 3. Index Constituents File (optional)

Supported filenames:

* `index_constituents_long.csv`
* `nse_all_index_constituents.csv`

Requires:

* Symbol column (e.g., `Symbol`, `Stock`)
* Index name column (e.g., `Index`, `IndexName`)

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/stock-dashboard
cd stock-dashboard
```

### 2. Install dependencies

```
pip install streamlit pandas numpy plotly
```

### 3. Run the app

```
streamlit run app.py
```

---

## How It Works

### Select Mode (Sidebar)

* **Stock Mode:** choose a stock; charts highlight that stock.
* **Sector Mode:** choose an industry; charts display all stocks.
* **Index Mode:** load and analyze index constituents from CSV; unmatched symbols are shown.

### Charts Tab

* Configure X and Y metrics for each chart.
* Toggle outlier removal using IQR.
* Hover tooltips display detailed stock metrics.
* Download CSV for data behind each chart.

### Concall Tab

* Select a stock to read summary text.
* Keyword search across all concall summaries.

---

## Troubleshooting

### No numeric columns found

Your `stocks_info.csv` is missing numeric metric fields.

### Index Mode not working

Index CSV missing symbol or index name columns.

### Concall summaries not loading

Ensure correct filename and a valid symbol column.

---

## License

MIT License (or add your own).

---

## Contribution

Feel free to open issues or submit pull requests to improve features, performance, or documentation.

# stockPatternDetector# Stock Pattern Detector & Short-Term Analysis

This Streamlit app allows you to analyze stocks for bullish candlestick patterns, key technical indicators, and short-term trading insights using data from Yahoo Finance. It provides interactive charts with overlays for EMAs, MACD, RSI, volume, and detected candlestick patterns, along with rules-based analysis for volume, moving averages, and market stage.

## Features

- **Fetch Historical Stock Data:** Retrieve OHLCV data for any ticker using Yahoo Finance.
- **Candlestick Pattern Detection:** Automatically detects common bullish and bearish candlestick patterns using TA-Lib.
- **Technical Indicators:** Calculates EMAs (10, 20, 50, 200), MACD, RSI, 20-day average volume, and RVOL.
- **Rules-Based Analysis:** Provides short-term trading insights based on volume confirmation/divergence, EMA trends/crossovers, and Stan Weinstein's Stage Analysis.
- **Interactive Visualization:** Plots candlestick charts with overlays for indicators and pattern markers using mplfinance.
- **Streamlit UI:** User-friendly interface to input ticker and duration, view insights, and interact with the chart.

## Usage

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   > **Note:** TA-Lib may require additional system dependencies. See [TA-Lib installation instructions](https://mrjbq7.github.io/ta-lib/install.html).

2. **Run the app:**
   ```
   streamlit run main.py
   ```

3. **Interact:**
   - Enter a stock ticker (e.g., `AAPL`).
   - Select a historical duration (e.g., `6mo`, `1y`, etc.).
   - Click "Analyze Stock" to view detected patterns, technical indicators, and trading insights.

## File Structure

- [`main.py`](main.py): Main Streamlit application with all logic and UI.
- [`requirements.txt`](requirements.txt): List of required Python packages.

## Example Screenshot

![screenshot](https://user-images.githubusercontent.com/yourusername/yourrepo/screenshot.png)

## Requirements

- Python 3.7+
- See [`requirements.txt`](requirements.txt) for Python dependencies.

## License

MIT License

---

**Disclaimer:** This tool is for educational and informational purposes only. It is not financial advice. Always do your own research before making investment decisions.
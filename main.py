import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import talib 
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta

# --- Your existing stock analysis functions (copied for convenience) ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent re-fetching on every interaction
def fetch_historical_data(ticker_symbol, period='6mo'):
    """
    Fetches historical stock data for a given ticker symbol.
    """
    try:
        data = yf.download(ticker_symbol, period=period)
        if data.empty:
            st.error(f"No data found for {ticker_symbol} for the period {period}. Check ticker or period.")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            new_columns = [col[0].capitalize() for col in data.columns]
            data.columns = new_columns
        else:
            data.columns = [col.capitalize() for col in data.columns]

        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        if 'Adj Close' in data.columns and 'Close' in data.columns and not data['Close'].equals(data['Adj Close']):
             st.warning("Both 'Close' and 'Adj Close' present. Using 'Close' for patterns and plotting.")
             data.drop(columns=['Adj Close'], inplace=True, errors='ignore')

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Missing one or more required columns in fetched data: {required_cols}. Available: {data.columns.tolist()}")
            return None
        
        data.dropna(subset=required_cols, inplace=True)
        
        if data.empty:
            st.warning(f"No valid data after dropping NaNs for {ticker_symbol}.")
            return None

        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(subset=required_cols, inplace=True)

        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def detect_bullish_patterns(df):
    """
    Detects common bullish candlestick patterns in the DataFrame.
    """
    if df is None or df.empty:
        return None

    required_ohlc = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_ohlc):
        st.error(f"Missing OHLC columns for pattern detection: {required_ohlc}")
        return df.copy()

    op = df['Open'].values
    hi = df['High'].values
    lo = df['Low'].values
    cl = df['Close'].values

    if len(op) == 0:
        st.info("OHLC arrays are empty, cannot detect patterns.")
        return df.copy()

    df_result = df.copy()

    try:
        df_result['CDLHAMMER'] = talib.CDLHAMMER(op, hi, lo, cl)
        df_result['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(op, hi, lo, cl)
        df_result['CDLBULLISHENGULFING'] = talib.CDLBULLISHENGULFING(op, hi, lo, cl)
        df_result['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(op, hi, lo, cl)
        df_result['CDLPIERCING'] = talib.CDLPIERCING(op, hi, lo, cl)
        df_result['CDLTHREEWHITESOLDIERS'] = talib.CDLTHREEWHITESOLDIERS(op, hi, lo, cl)
        df_result['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(op, hi, lo, cl)
        df_result['CDLHARAMI'] = talib.CDLHARAMI(op, hi, lo, cl)
        
        # Adding bearish patterns for completeness in general analysis, though focusing on bullish signals later
        df_result['CDLDOJI'] = talib.CDLDOJI(op, hi, lo, cl)
        df_result['CDLBEARISHENGULFING'] = talib.CDLENGULFING(op, hi, lo, cl) # -100 for bearish engulfing
        df_result['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(op, hi, lo, cl)
        df_result['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(op, hi, lo, cl)
        df_result['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)
        df_result['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(op, hi, lo, cl)


        pattern_columns = [
            'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLBULLISHENGULFING',
            'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLTHREEWHITESOLDIERS',
            'CDLDRAGONFLYDOJI', 'CDLHARAMI', 'CDLDOJI', 'CDLBEARISHENGULFING',
            'CDLDARKCLOUDCOVER', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN'
        ]

        for col in pattern_columns:
            # For engulfing, bullish is 100, bearish is -100.
            # For others, 100 for bullish, -100 for bearish. Doji is 100.
            if col == 'CDLBEARISHENGULFING':
                df_result[f'{col}_Detected'] = (df_result[col] == -100).fillna(False)
            elif col == 'CDLDOJI':
                df_result[f'{col}_Detected'] = (df_result[col] != 0).fillna(False) # Doji is 100 for detection
            else:
                df_result[f'{col}_Detected'] = (df_result[col] == 100).fillna(False) 
            df_result = df_result.drop(columns=[col])

    except Exception as e:
        st.error(f"Error detecting patterns with TA-Lib: {e}")
        st.info("Please ensure TA-Lib is correctly installed (check console for details).")
        return df_result

    return df_result

# --- Calculate Key Indicators ---
def calculate_key_indicators(df):
    """
    Calculates various key technical indicators and adds them to the DataFrame.
    """
    if df is None or df.empty:
        return None

    if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
        st.warning("Close price data not available or not numeric for indicator calculation.")
        return df
    if 'Volume' not in df.columns or not pd.api.types.is_numeric_dtype(df['Volume']):
        st.warning("Volume data not available or not numeric for indicator calculation.")
        pass

    close_prices = df['Close'].values
    volume_data = df['Volume'].values

    for period in [10, 20, 50, 200]:
        try:
            if len(close_prices) >= period:
                df[f'EMA_{period}'] = talib.EMA(close_prices, timeperiod=period)
            else:
                df[f'EMA_{period}'] = np.nan 
                st.warning(f"Not enough data ({len(close_prices)} days) to calculate EMA_{period}. Required: {period} days.")
        except Exception as e:
            df[f'EMA_{period}'] = np.nan
            st.warning(f"Could not calculate EMA_{period} due to an error: {e}")

    if 'Volume' in df.columns:
        try:
            # Calculate 20-day volume average
            if len(df) >= 20: 
                df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
            else:
                df['Volume_Avg_20'] = np.nan
                st.warning(f"Not enough data ({len(df)} days) to calculate 20-day Volume Average. Required: 20 days.")

            # Calculate Relative Volume (RVOL)
            if 'Volume_Avg_20' in df.columns and not df['Volume_Avg_20'].isnull().all():
                df['RVOL'] = df['Volume'] / df['Volume_Avg_20']
            else:
                df['RVOL'] = np.nan

        except Exception as e:
            df['Volume_Avg_20'] = np.nan
            df['RVOL'] = np.nan
            st.warning(f"Could not calculate Volume Average or RVOL due to an error: {e}")

    if len(close_prices) >= 34: 
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                                      fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist
        except Exception as e:
            df['MACD'] = np.nan; df['MACD_Signal'] = np.nan; df['MACD_Hist'] = np.nan
            st.warning(f"Could not calculate MACD due to an error: {e}")
    else:
        df['MACD'] = np.nan; df['MACD_Signal'] = np.nan; df['MACD_Hist'] = np.nan
        st.warning(f"Not enough data ({len(close_prices)} days) to calculate MACD. Required: ~34 days.")

    if len(close_prices) >= 14:
        try:
            df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        except Exception as e:
            df['RSI'] = np.nan
            st.warning(f"Could not calculate RSI due to an error: {e}")
    else:
        df['RSI'] = np.nan
        st.warning(f"Not enough data ({len(close_prices)} days) to calculate RSI. Required: 14 days.")

    return df

# --- Plotting function updated to fix panel overlapping ---
def plot_stock_with_patterns(df, ticker_symbol):
    """
    Plots the stock candlestick chart, overlays detected bullish patterns,
    and includes key indicators. Returns the mplfinance figure object.
    """
    if df is None or df.empty:
        st.info("No data to plot as DataFrame is empty or None.")
        return None

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"DataFrame for plotting is missing required OHLCV columns: {required_cols}")
        st.dataframe(df.head())
        st.write(df.dtypes)
        return None
    
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' is not numeric. Cannot plot.")
            st.dataframe(df.head())
            st.write(df.dtypes)
            return None

    apds = [] 
    
    # Define panel ratios based on expected panels.
    # We will build this list dynamically as well.
    # Base panels: Price (0), Volume (1)
    panel_ratios_config = [6, 2] # Default for price and volume
    
    # Track the current panel index for new indicators
    current_panel_index = 2 # MACD will try for panel 2, RSI for panel 3 (if both active)

    # --- Add EMA lines to the main panel (panel=0) ---
    ema_colors = {10: 'blue', 20: 'orange', 50: 'purple', 200: 'red'}
    for period in [10, 20, 50, 200]:
        ema_col_name = f'EMA_{period}'
        if ema_col_name in df.columns and not df[ema_col_name].isnull().all():
            apds.append(
                mpf.make_addplot(df[ema_col_name], panel=0, color=ema_colors.get(period, 'gray'), width=0.8, linestyle='-', title=f'EMA{period}')
            )

    # --- Add Volume Average to the volume panel (panel=1) ---
    if 'Volume_Avg_20' in df.columns and not df['Volume_Avg_20'].isnull().all():
        apds.append(
            mpf.make_addplot(df['Volume_Avg_20'], panel=1, color='darkblue', linestyle='--', width=0.8, title='Vol Avg')
        )

    # --- Add MACD to a new panel (dynamically assigned index) ---
    # Check if MACD data is actually present and not all NaN
    has_macd_data = False
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns \
       and not df['MACD'].isnull().all():
        has_macd_data = True
        
        # Add MACD lines and histogram using the next available panel index
        apds.append(mpf.make_addplot(df['MACD'], panel=current_panel_index, color='blue', ylabel='MACD', title='MACD'))
        apds.append(mpf.make_addplot(df['MACD_Signal'], panel=current_panel_index, color='red'))
        
        colors = ['green' if x >= 0 else 'red' for x in df['MACD_Hist'].fillna(0)]
        apds.append(mpf.make_addplot(df['MACD_Hist'], panel=current_panel_index, type='bar', color=colors, width=0.7, alpha=0.6))
        
        panel_ratios_config.append(3) # Add ratio for MACD panel
        current_panel_index += 1 # Increment panel index for the next indicator

    # --- Add RSI to another new panel (dynamically assigned index) ---
    # Check if RSI data is actually present and not all NaN
    has_rsi_data = False
    if 'RSI' in df.columns and not df['RSI'].isnull().all():
        has_rsi_data = True
        
        # Add RSI lines using the next available panel index
        apds.append(mpf.make_addplot(df['RSI'], panel=current_panel_index, color='purple', ylabel='RSI', title='RSI'))
        apds.append(mpf.make_addplot(pd.Series(70, index=df.index), panel=current_panel_index, color='gray', linestyle='--', width=0.7))
        apds.append(mpf.make_addplot(pd.Series(30, index=df.index), panel=current_panel_index, color='gray', linestyle='--', width=0.7))
        
        panel_ratios_config.append(3) # Add ratio for RSI panel
        current_panel_index += 1 # Increment panel index (though not strictly needed after the last indicator)


    # --- Add Candlestick Pattern Markers (existing logic, still on panel=0) ---
    pattern_labels = {
        'CDLHAMMER_Detected': 'Hammer',
        'CDLINVERTEDHAMMER_Detected': 'Inv Hammer',
        'CDLBULLISHENGULFING_Detected': 'Bull Engulf',
        'CDLMORNINGSTAR_Detected': 'Morning Star',
        'CDLPIERCING_Detected': 'Piercing',
        'CDLTHREEWHITESOLDIERS_Detected': '3 White Soldiers',
        'CDLDRAGONFLYDOJI_Detected': 'Dragonfly Doji',
        'CDLHARAMI_Detected': 'Harami',
        'CDLDOJI_Detected': 'Doji',
        'CDLBEARISHENGULFING_Detected': 'Bear Engulf',
        'CDLDARKCLOUDCOVER_Detected': 'Dark Cloud',
        'CDLEVENINGSTAR_Detected': 'Evening Star',
        'CDLSHOOTINGSTAR_Detected': 'Shooting Star',
        'CDLHANGINGMAN_Detected': 'Hanging Man'
    }

    for col, label in pattern_labels.items():
        if col in df.columns and df[col].any():
            pattern_dates = df.index[df[col]].tolist()
            for date in pattern_dates:
                if date in df.index:
                    is_bullish = 'CDLHAMMER' in label or 'Bull Engulf' in label or 'Morning Star' in label or 'Piercing' in label or '3 White Soldiers' in label or 'Dragonfly Doji' in label or 'Harami' in label or 'Inv Hammer' in label
                    
                    marker_color = 'green' if is_bullish else 'red'
                    marker_type = '^' if is_bullish else 'v'
                    y_offset = 0.98 if is_bullish else 1.02 # Adjust marker position for visibility

                    price_point = df.loc[date, 'Low'] if is_bullish else df.loc[date, 'High']

                    apds.append(
                        mpf.make_addplot(
                            pd.Series(price_point * y_offset, index=[date]),
                            type='scatter',
                            marker=marker_type,
                            markersize=100,
                            color=marker_color,
                            panel=0, 
                        )
                    )
                    apds.append(
                        mpf.make_addplot(
                            pd.Series(price_point * (y_offset - 0.02 * (-1 if is_bullish else 1)), index=[date]), # Text offset
                            type='scatter',
                            marker='',
                            text=label,
                            fontscale=0.8,
                            color='blue', # Text color can be consistent
                            panel=0, 
                        )
                    )

    plot_kwargs = dict(
        type='candle',
        style='yahoo',
        volume=True,
        title=f"{ticker_symbol} Stock Chart with Patterns and Indicators",
        ylabel='Price',
        ylabel_lower='Volume',
        figscale=2.0, # Increased figure scale for more vertical space
        returnfig=True,
        # Use the dynamically built panel_ratios_config tuple
        panel_ratios=tuple(panel_ratios_config) 
    )

    if apds:
        plot_kwargs['addplot'] = apds

    try:
        fig, axlist = mpf.plot(df, **plot_kwargs)
        return fig
    except Exception as e:
        st.error(f"Error plotting with mplfinance: {e}")
        st.info("Ensure the data has sufficient length and valid numeric types for plotting. Check console for details.")
        return None


# --- New Functions for Rules-Based Analysis ---

def analyze_volume(df, lookback_period=5):
    """
    Analyzes recent volume for confirmation, divergence, and spikes.
    Returns a dictionary of volume insights.
    """
    volume_insights = []

    if df is None or df.empty or 'Volume' not in df.columns or 'RVOL' not in df.columns:
        volume_insights.append("Insufficient data for volume analysis.")
        return volume_insights

    latest_data = df.iloc[-lookback_period:]
    if len(latest_data) < lookback_period:
        volume_insights.append(f"Not enough recent data for {lookback_period}-day volume analysis.")
        return volume_insights

    current_volume = df['Volume'].iloc[-1]
    current_rvol = df['RVOL'].iloc[-1]
    current_close = df['Close'].iloc[-1]
    previous_close = df['Close'].iloc[-2] if len(df) >= 2 else np.nan

    # Volume Confirmation
    if not np.isnan(previous_close):
        if current_volume > df['Volume_Avg_20'].iloc[-1]: # Check against average for significance
            if current_close > previous_close:
                volume_insights.append(f"🟢 **Bullish Volume Confirmation**: Price rising on higher than average volume (RVOL: {current_rvol:.2f}).")
            elif current_close < previous_close:
                volume_insights.append(f"🔴 **Bearish Volume Confirmation**: Price falling on higher than average volume (RVOL: {current_rvol:.2f}).")
        else:
            if current_close > previous_close:
                volume_insights.append(f"🟡 **Weak Bullish Move**: Price rising but on lower than average volume (RVOL: {current_rvol:.2f}).")
            elif current_close < previous_close:
                volume_insights.append(f"🟡 **Weak Bearish Move**: Price falling but on lower than average volume (RVOL: {current_rvol:.2f}).")

    # Volume Spikes (e.g., RVOL > 2.0 or 3.0)
    if current_rvol >= 2.0:
        volume_insights.append(f"⚡ **Volume Spike Detected**: RVOL at {current_rvol:.2f}. Indicates unusual activity.")
    elif current_rvol >= 1.5:
        volume_insights.append(f"📈 **Elevated Volume**: RVOL at {current_rvol:.2f}. Higher than usual activity.")

    # Volume Divergence (simplified for recent data)
    # Check if price made a new high/low but volume didn't confirm
    if len(latest_data) >= 3: # Need at least 3 days to compare trends
        recent_prices = latest_data['Close']
        recent_volumes = latest_data['Volume']

        if recent_prices.iloc[-1] > recent_prices.iloc[-2] and recent_prices.iloc[-2] > recent_prices.iloc[-3] and \
           recent_volumes.iloc[-1] < recent_volumes.iloc[-2]:
            volume_insights.append("⚠️ **Bearish Volume Divergence (Price Up, Vol Down)**: Recent price highs not confirmed by increasing volume.")
        elif recent_prices.iloc[-1] < recent_prices.iloc[-2] and recent_prices.iloc[-2] < recent_prices.iloc[-3] and \
             recent_volumes.iloc[-1] < recent_volumes.iloc[-2]:
            volume_insights.append("⚠️ **Bullish Volume Divergence (Price Down, Vol Down)**: Recent price lows not confirmed by increasing volume (could signal exhaustion).")

    return volume_insights

def analyze_emas(df):
    """
    Analyzes EMA trends, crossovers, and price relationships.
    Returns a dictionary of EMA insights.
    """
    ema_insights = []

    if df is None or df.empty:
        ema_insights.append("Insufficient data for EMA analysis.")
        return ema_insights
    
    # Ensure EMAs are calculated
    for period in [10, 20, 50, 200]:
        if f'EMA_{period}' not in df.columns or df[f'EMA_{period}'].isnull().all():
            ema_insights.append(f"EMA_{period} not available for analysis.")

    # Check if at least 2 days of data are available for slope and crossovers
    if len(df) < 2:
        ema_insights.append("Not enough data to analyze EMA slopes or crossovers.")
        return ema_insights

    latest_close = df['Close'].iloc[-1]
    
    # EMA Trend Identification (slope of current EMA)
    for period in [10, 20, 50]: # Short-term to medium-term EMAs
        ema_col = f'EMA_{period}'
        if ema_col in df.columns and not pd.isna(df[ema_col].iloc[-1]) and not pd.isna(df[ema_col].iloc[-2]):
            current_ema = df[ema_col].iloc[-1]
            prev_ema = df[ema_col].iloc[-2]
            
            if current_ema > prev_ema:
                ema_insights.append(f"📈 **EMA_{period} Trend**: Uptrend (EMA is rising).")
            elif current_ema < prev_ema:
                ema_insights.append(f"📉 **EMA_{period} Trend**: Downtrend (EMA is falling).")
            else:
                ema_insights.append(f"↔️ **EMA_{period} Trend**: Sideways (EMA is flat).")

            # Price vs. EMA
            if latest_close > current_ema:
                ema_insights.append(f"✅ Price is above EMA_{period} (bullish sign).")
            elif latest_close < current_ema:
                ema_insights.append(f"❌ Price is below EMA_{period} (bearish sign).")
            else:
                ema_insights.append(f"⚖️ Price is at EMA_{period}.")

    # EMA Crossovers
    # 10-day EMA and 20-day EMA crossover
    if 'EMA_10' in df.columns and 'EMA_20' in df.columns and len(df) >= 2:
        ema10_current = df['EMA_10'].iloc[-1]
        ema20_current = df['EMA_20'].iloc[-1]
        ema10_prev = df['EMA_10'].iloc[-2]
        ema20_prev = df['EMA_20'].iloc[-2]

        if not (pd.isna(ema10_current) or pd.isna(ema20_current) or pd.isna(ema10_prev) or pd.isna(ema20_prev)):
            if ema10_current > ema20_current and ema10_prev <= ema20_prev:
                ema_insights.append("⭐ **Bullish Crossover (EMA10 above EMA20)**: Short-term momentum gaining.")
            elif ema10_current < ema20_current and ema10_prev >= ema20_prev:
                ema_insights.append("❗ **Bearish Crossover (EMA10 below EMA20)**: Short-term momentum weakening.")
    
    # 50-day EMA and 200-day EMA crossover (Golden/Death Cross - medium/long term, but good to know)
    if 'EMA_50' in df.columns and 'EMA_200' in df.columns and len(df) >= 2:
        ema50_current = df['EMA_50'].iloc[-1]
        ema200_current = df['EMA_200'].iloc[-1]
        ema50_prev = df['EMA_50'].iloc[-2]
        ema200_prev = df['EMA_200'].iloc[-2]

        if not (pd.isna(ema50_current) or pd.isna(ema200_current) or pd.isna(ema50_prev) or pd.isna(ema200_prev)):
            if ema50_current > ema200_current and ema50_prev <= ema200_prev:
                ema_insights.append("🌟 **Golden Cross (EMA50 above EMA200)**: Long-term bullish signal.")
            elif ema50_current < ema200_current and ema50_prev >= ema200_prev:
                ema_insights.append("💀 **Death Cross (EMA50 below EMA200)**: Long-term bearish signal.")

    return ema_insights

def analyze_stage(df):
    """
    Applies a simplified Stan Weinstein's Stage Analysis using EMAs.
    Focus on EMA_50 (for medium term) and EMA_200 (for long term).
    Requires a decent amount of historical data (e.g., 1 year minimum for 200 EMA).
    Returns a string indicating the current stage.
    """
    if df is None or df.empty:
        return "Insufficient data for Stage Analysis."
    
    # We need at least 200 data points for EMA_200
    if len(df) < 200:
        return f"Not enough data ({len(df)} days) for comprehensive Stage Analysis (needs ~200 days for EMA_200)."

    # Get the most recent values
    close = df['Close'].iloc[-1]
    ema_50 = df['EMA_50'].iloc[-1]
    ema_200 = df['EMA_200'].iloc[-1]

    # Check for NaN values in EMAs before proceeding
    if pd.isna(ema_50) or pd.isna(ema_200):
        return "EMA_50 or EMA_200 data missing for Stage Analysis."
    
    # EMA slopes (simplified: compare current to a few days prior)
    # Using a 5-day lookback for slope to smooth out daily noise
    ema_50_slope = (df['EMA_50'].iloc[-1] - df['EMA_50'].iloc[-5]) if len(df) >= 5 and not pd.isna(df['EMA_50'].iloc[-5]) else 0
    ema_200_slope = (df['EMA_200'].iloc[-1] - df['EMA_200'].iloc[-5]) if len(df) >= 5 and not pd.isna(df['EMA_200'].iloc[-5]) else 0

    # Stage 2 criteria (Advancing/Markup):
    # Price is above EMA_50 and EMA_200.
    # EMA_50 is above EMA_200.
    # Both EMAs are trending upwards.
    if close > ema_50 and close > ema_200 and \
       ema_50 > ema_200 and \
       ema_50_slope > 0 and ema_200_slope > 0:
        return "🟢 **Stage 2 (Advancing/Markup)**: Ideal for long positions. Price is trending up, above rising EMAs."

    # Stage 4 criteria (Declining/Downtrend):
    # Price is below EMA_50 and EMA_200.
    # EMA_50 is below EMA_200.
    # Both EMAs are trending downwards.
    if close < ema_50 and close < ema_200 and \
       ema_50 < ema_200 and \
       ema_50_slope < 0 and ema_200_slope < 0:
        return "🔴 **Stage 4 (Declining/Downtrend)**: Avoid long positions. Price is trending down, below falling EMAs."

    # Stage 1 criteria (Basing/Accumulation):
    # Price moves sideways, often after a decline. EMA_200 flattens or turns slightly up.
    # Price might be crossing above/below EMA_50, but generally below EMA_200 or near it.
    # EMA_50 might be below EMA_200, but flattening or turning up.
    # This is a bit more nuanced. We'll approximate by checking if it's not Stage 2 or 4,
    # and price is near EMA_200, and EMA_200 is relatively flat or turning up.
    # Or price is crossing EMA_200 from below.
    if (abs(close - ema_200) / ema_200 < 0.05) and (ema_200_slope >= -0.001) and (ema_50_slope >= -0.001): # Within 5% of EMA_200, and flat/upward sloping EMAs
        if not (close > ema_50 and ema_50 > ema_200 and ema_50_slope > 0): # Not a clear Stage 2
            return "🟡 **Stage 1 (Basing/Accumulation)**: Price consolidating, EMAs flattening. Potential for future breakout."

    # Stage 3 criteria (Topping/Distribution):
    # Upward momentum slows. Price becomes erratic, often moving sideways or showing signs of weakness.
    # Price might break below EMA_50, EMA_50 might flatten or cross below EMA_200.
    # We'll approximate by checking if it's not Stage 1, 2, or 4, and EMA_50 is flattening or turning down.
    if (close > ema_200 and ema_50 > ema_200) and (ema_50_slope <= 0.001): # Price above 200, 50 above 200, but 50 flattening/down
        if not (ema_50_slope > 0.001 and ema_200_slope > 0.001): # Not a strong Stage 2 trend
            return "🟠 **Stage 3 (Topping/Distribution)**: Upward momentum slowing, potential reversal. Smart money might be selling."
    
    # Fallback if no clear stage is identified
    return "⚪ **Unclear Stage**: Current price action does not fit a defined Weinstein Stage clearly."


# --- Streamlit UI Layout (no changes needed here) ---

st.set_page_config(layout="wide", page_title="Stock Pattern Detector & Short-Term Analysis")

st.title("📈 Stock Pattern Detector & Short-Term Analysis")
st.markdown("Enter a stock ticker and select a historical duration to analyze for bullish candlestick patterns and view key technical indicators, along with short-term trading insights.")

col1, col2 = st.columns(2)

with col1:
    ticker_symbol = st.text_input("Stock Ticker Symbol", "AAPL").strip().upper()

with col2:
    duration_options = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    duration = st.selectbox("Historical Duration", duration_options, index=duration_options.index('6mo'))

if st.button("Analyze Stock"):
    if not ticker_symbol:
        st.error("Please enter a stock ticker symbol.")
    else:
        with st.spinner(f"Analyzing {ticker_symbol} for Patterns & Short-Term Insights (last {duration})..."):
            df = fetch_historical_data(ticker_symbol, period=duration)
            if df is None:
                st.stop()

            if df.empty or len(df) < 1:
                st.info(f"Fetched data for {ticker_symbol} is too short ({len(df)} days) to display a meaningful chart or perform analysis. Please select a longer duration.")
                st.stop()

            df_with_indicators = calculate_key_indicators(df.copy())
            if df_with_indicators is None:
                st.error("Failed to calculate indicators.")
                st.stop()

            df_final = detect_bullish_patterns(df_with_indicators.copy())

            if df_final is not None:
                st.subheader(f"Short-Term Trading Insights for {ticker_symbol}")
                
                # --- Short-Term Outlook Section ---
                st.markdown("---")
                st.markdown("### 📊 Short-Term Trading Outlook:")
                
                # 1. Volume Analysis
                st.markdown("#### Volume Analysis:")
                volume_insights = analyze_volume(df_final)
                if volume_insights:
                    for insight in volume_insights:
                        st.markdown(insight)
                else:
                    st.info("No specific volume insights to report.")

                # 2. EMA Analysis
                st.markdown("#### Exponential Moving Average (EMA) Analysis:")
                ema_insights = analyze_emas(df_final)
                if ema_insights:
                    for insight in ema_insights:
                        st.markdown(insight)
                else:
                    st.info("No specific EMA insights to report.")

                # 3. Stan Weinstein's Stage Analysis
                st.markdown("#### Stan Weinstein's Stage Analysis (Simplified):")
                stage_analysis_result = analyze_stage(df_final)
                st.markdown(stage_analysis_result)


                # --- Candlestick Pattern Summary Section ---
                st.markdown("---")
                st.subheader("Detected Candlestick Patterns:")
                detected_any_pattern = False
                pattern_summary_details = []
                for col in df_final.columns:
                    if '_Detected' in col:
                        num_detections = df_final[col].sum()
                        if num_detections > 0:
                            detected_any_pattern = True
                            pattern_name = col.replace('_Detected', '')
                            dates = df_final.index[df_final[col]].strftime('%Y-%m-%d').tolist()
                            pattern_summary_details.append(f"- **{pattern_name}**: {num_detections} occurrences on: {', '.join(dates)}")
                
                if not detected_any_pattern:
                    st.info("No significant candlestick patterns detected in the specified period.")
                else:
                    for detail in pattern_summary_details:
                        st.markdown(detail)
                    
                st.success("Analysis complete. Chart displayed below.")
                st.markdown("---")


                # --- Plotting the chart ---
                fig = plot_stock_with_patterns(df_final, ticker_symbol)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("Failed to generate plot.")
            else:
                st.error("Failed to detect patterns.")
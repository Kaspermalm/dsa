#!/usr/bin/env python3
"""
Streamlit Dashboard for DSA Prediction Monitoring.
Displays real-time predictions, historical accuracy, and feature insights.

Run with: streamlit run src/dashboard.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import streamlit as st
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
    import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.express as px
    import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

# Import config with fallback for Streamlit Cloud
try:
    from config import COMBINED_CSV, PREDICTIONS_CSV
except ImportError:
    # Fallback paths for Streamlit Cloud
    COMBINED_CSV = "csv/dsa_news_combined.csv"
    PREDICTIONS_CSV = "csv/predictions_history.csv"

# Hindcast files
HINDCAST_CSV = "csv/hindcast_results.csv"
HINDCAST_METRICS_CSV = "csv/hindcast_metrics.csv"


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="DSA Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_historical_data():
    """Load historical DSA + news data."""
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    return pd.DataFrame()


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_predictions():
    """Load prediction history."""
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
        df['target_date'] = pd.to_datetime(df['target_date'])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        return df.sort_values('target_date', ascending=False)
    return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour (hindcast doesn't change often)
def load_hindcast_results():
    """Load hindcast backtesting results."""
    if os.path.exists(HINDCAST_CSV):
        df = pd.read_csv(HINDCAST_CSV)
        df['target_date'] = pd.to_datetime(df['target_date'])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_hindcast_metrics():
    """Load hindcast summary metrics."""
    if os.path.exists(HINDCAST_METRICS_CSV):
        return pd.read_csv(HINDCAST_METRICS_CSV)
    return pd.DataFrame()


def try_load_from_hopsworks():
    """Try to load data from Hopsworks Feature Store."""
    try:
        from hopsworks_utils import get_feature_store, get_prediction_history
        fs = get_feature_store()
        
        # Get predictions from Hopsworks
        pred_df = get_prediction_history(fs, limit=100)
        
        # Filter out sentinel values (-1 means no actual value yet)
        if not pred_df.empty and 'actual' in pred_df.columns:
            pred_df.loc[pred_df['actual'] == -1, 'actual'] = None
        
        # Get historical data from local CSV (more reliable)
        hist_df = load_historical_data()
        
        return hist_df, pred_df, True
    except Exception as e:
        st.error(f"Hopsworks error: {e}")
        return None, None, False


# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

def main():
    st.title("📊 DSA Content Moderation Prediction Dashboard")
    st.markdown("Monitoring EU Digital Services Act violations and prediction accuracy")
    
    # Sidebar
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio(
        "Data Source",
        ["Local CSV", "Hopsworks Feature Store"],
        index=0
    )
    
    days_to_show = st.sidebar.slider("Days to display", 7, 365, 90)
    
    # Load data based on selection
    if data_source == "Hopsworks Feature Store":
        hist_df, pred_df, success = try_load_from_hopsworks()
        if not success:
            st.warning("Could not connect to Hopsworks. Falling back to local CSV.")
            hist_df = load_historical_data()
            pred_df = load_predictions()
    else:
        hist_df = load_historical_data()
        pred_df = load_predictions()
    
    if hist_df.empty:
        st.error("No historical data found. Run `make backfill` first.")
        return
    
    # Filter to recent data
    cutoff = datetime.now() - timedelta(days=days_to_show)
    hist_df = hist_df[hist_df['date'] >= cutoff]
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_count = hist_df.iloc[-1]['dsa_count'] if len(hist_df) > 0 else 0
        prev_count = hist_df.iloc[-2]['dsa_count'] if len(hist_df) > 1 else latest_count
        delta = latest_count - prev_count
        st.metric(
            "Latest DSA Violations",
            f"{latest_count:,.0f}",
            delta=f"{delta:+,.0f}",
            delta_color="inverse"
        )
    
    with col2:
        avg_7d = hist_df.tail(7)['dsa_count'].mean() if len(hist_df) >= 7 else 0
        st.metric("7-Day Average", f"{avg_7d:,.0f}")
    
    with col3:
        avg_sentiment = hist_df['sentiment_mean'].mean() if 'sentiment_mean' in hist_df.columns else 0
        st.metric(
            "Avg News Sentiment",
            f"{avg_sentiment:.3f}",
            delta="Neutral" if abs(avg_sentiment) < 0.1 else ("Positive" if avg_sentiment > 0 else "Negative")
        )
    
    with col4:
        if not pred_df.empty and 'actual' in pred_df.columns:
            evaluated = pred_df[pred_df['actual'].notna()]
            if len(evaluated) > 0:
                mape = evaluated['pct_error'].mean() if 'pct_error' in evaluated.columns else 0
                st.metric("Prediction MAPE", f"{mape:.1f}%")
            else:
                st.metric("Prediction MAPE", "N/A")
        else:
            st.metric("Prediction MAPE", "N/A")
    
    # ========================================================================
    # HISTORICAL TREND
    # ========================================================================
    
    st.header("📉 Historical DSA Violations")
    
    fig = go.Figure()
    
    # Daily counts
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['dsa_count'],
        mode='lines',
        name='Daily Count',
        line=dict(color='#3498db', width=1),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    # 7-day rolling average
    rolling_avg = hist_df['dsa_count'].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=rolling_avg,
        mode='lines',
        name='7-Day Average',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="DSA Violation Count",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PREDICTIONS
    # ========================================================================
    
    st.header("🔮 Predictions")
    
    if not pred_df.empty:
        # Get future predictions (not yet evaluated)
        future_preds = pred_df[pred_df['actual'].isna()].head(7)
        
        if len(future_preds) > 0:
            st.subheader("Upcoming Forecasts")
            
            cols = st.columns(min(7, len(future_preds)))
            for i, (_, row) in enumerate(future_preds.iterrows()):
                with cols[i]:
                    date_str = row['target_date'].strftime('%b %d')
                    day_name = row['target_date'].strftime('%a')
                    st.metric(
                        f"{day_name} {date_str}",
                        f"{row['predicted']:,.0f}"
                    )
        
        # Prediction accuracy
        evaluated = pred_df[pred_df['actual'].notna()]
        
        if len(evaluated) > 0:
            st.subheader("Prediction Accuracy")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=evaluated['target_date'],
                y=evaluated['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=evaluated['target_date'],
                y=evaluated['predicted'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="DSA Count",
                hovermode='x unified',
                height=350
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No predictions available. Run `make infer` to generate forecasts.")
    
    # ========================================================================
    # NEWS SENTIMENT
    # ========================================================================
    
    if 'sentiment_mean' in hist_df.columns:
        st.header("📰 News Sentiment")
        
        fig3 = go.Figure()
        
        colors = ['#27ae60' if s > 0 else '#e74c3c' for s in hist_df['sentiment_mean']]
        
        fig3.add_trace(go.Bar(
            x=hist_df['date'],
            y=hist_df['sentiment_mean'],
            marker_color=colors,
            name='Sentiment'
        ))
        
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Sentiment (-1 to 1)",
            height=300
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # ========================================================================
    # HINDCAST (BACKTESTING) RESULTS
    # ========================================================================
    
    hindcast_df = load_hindcast_results()
    hindcast_metrics = load_hindcast_metrics()
    
    if not hindcast_df.empty and not hindcast_metrics.empty:
        st.header("🔬 Model Backtesting (Hindcast)")
        st.markdown("""
        *Walk-forward validation: Train on historical data, predict future days, compare to actuals.*
        """)
        
        # Get overall metrics
        overall = hindcast_metrics[hindcast_metrics['horizon'] == 'Overall']
        if len(overall) > 0:
            overall = overall.iloc[0]
            
            # Key hindcast metrics
            hc_col1, hc_col2, hc_col3, hc_col4 = st.columns(4)
            
            with hc_col1:
                st.metric(
                    "Backtest R²",
                    f"{overall['r2']:.3f}",
                    help="R-squared score across all hindcast predictions"
                )
            
            with hc_col2:
                st.metric(
                    "Backtest MAPE",
                    f"{overall['mape']:.1f}%",
                    help="Mean Absolute Percentage Error"
                )
            
            with hc_col3:
                st.metric(
                    "Within 20% Error",
                    f"{overall['pct_under_20']:.1f}%",
                    help="Percentage of predictions within 20% of actual"
                )
            
            with hc_col4:
                st.metric(
                    "Total Predictions",
                    f"{int(overall['count']):,}",
                    help="Number of hindcast predictions evaluated"
                )
        
        # Tabs for different hindcast visualizations
        hc_tab1, hc_tab2, hc_tab3 = st.tabs(["📊 Accuracy by Horizon", "📈 Actual vs Predicted", "📉 Error Distribution"])
        
        with hc_tab1:
            # MAPE by forecast horizon
            horizon_metrics = hindcast_metrics[hindcast_metrics['horizon'] != 'Overall'].copy()
            horizon_metrics['day'] = horizon_metrics['horizon'].str.extract(r'(\d+)').astype(int)
            horizon_metrics = horizon_metrics.sort_values('day')
            
            fig_horizon = go.Figure()
            
            # MAPE bars
            fig_horizon.add_trace(go.Bar(
                x=horizon_metrics['day'],
                y=horizon_metrics['mape'],
                name='MAPE (%)',
                marker_color='#3498db',
                text=[f"{v:.1f}%" for v in horizon_metrics['mape']],
                textposition='outside'
            ))
            
            fig_horizon.update_layout(
                title="Prediction Accuracy Degrades with Longer Horizons",
                xaxis_title="Forecast Horizon (Days Ahead)",
                yaxis_title="Mean Absolute Percentage Error (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_horizon, use_container_width=True)
            
            # Also show R² by horizon
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=horizon_metrics['day'],
                y=horizon_metrics['r2'],
                mode='lines+markers',
                name='R²',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=10)
            ))
            
            fig_r2.update_layout(
                title="R² Score by Forecast Horizon",
                xaxis_title="Forecast Horizon (Days Ahead)",
                yaxis_title="R² Score",
                height=300,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with hc_tab2:
            # Scatter plot of actual vs predicted
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=hindcast_df['actual'],
                y=hindcast_df['predicted'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=hindcast_df['horizon'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title='Horizon')
                ),
                text=[f"Day {h}" for h in hindcast_df['horizon']],
                hovertemplate='Actual: %{x:,.0f}<br>Predicted: %{y:,.0f}<br>%{text}<extra></extra>'
            ))
            
            # Perfect prediction line
            min_val = min(hindcast_df['actual'].min(), hindcast_df['predicted'].min())
            max_val = max(hindcast_df['actual'].max(), hindcast_df['predicted'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            r2 = overall['r2'] if len(overall) > 0 else 0
            fig_scatter.update_layout(
                title=f"Hindcast: Actual vs Predicted (R² = {r2:.3f})",
                xaxis_title="Actual DSA Violations",
                yaxis_title="Predicted DSA Violations",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with hc_tab3:
            # Error distribution
            fig_error = go.Figure()
            
            fig_error.add_trace(go.Histogram(
                x=hindcast_df['pct_error'],
                nbinsx=50,
                marker_color='#3498db',
                name='Percentage Error'
            ))
            
            mean_error = hindcast_df['pct_error'].mean()
            median_error = hindcast_df['pct_error'].median()
            
            fig_error.add_vline(x=mean_error, line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {mean_error:.1f}%")
            fig_error.add_vline(x=median_error, line_dash="dash", line_color="orange",
                               annotation_text=f"Median: {median_error:.1f}%")
            
            fig_error.update_layout(
                title="Distribution of Prediction Errors",
                xaxis_title="Percentage Error (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
            
            # Summary stats
            st.markdown("**Error Breakdown:**")
            err_col1, err_col2, err_col3 = st.columns(3)
            with err_col1:
                pct_10 = (hindcast_df['pct_error'] < 10).mean() * 100
                st.metric("Within 10% error", f"{pct_10:.1f}%")
            with err_col2:
                pct_20 = (hindcast_df['pct_error'] < 20).mean() * 100
                st.metric("Within 20% error", f"{pct_20:.1f}%")
            with err_col3:
                pct_50 = (hindcast_df['pct_error'] < 50).mean() * 100
                st.metric("Within 50% error", f"{pct_50:.1f}%")
    else:
        st.header("🔬 Model Backtesting (Hindcast)")
        st.info("No hindcast data available. Run `make hindcast` to generate backtesting results.")
    
    # ========================================================================
    # DATA TABLE
    # ========================================================================
    
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            hist_df.tail(30).sort_values('date', ascending=False),
            use_container_width=True
        )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Data range: {hist_df['date'].min().strftime('%Y-%m-%d')} to {hist_df['date'].max().strftime('%Y-%m-%d')}*"
    )


if __name__ == "__main__":
    main()

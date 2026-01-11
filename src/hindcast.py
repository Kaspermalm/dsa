#!/usr/bin/env python3
"""
Hindcast (Backtesting) for DSA Prediction Model.

Performs walk-forward validation to evaluate model performance:
- Train on data up to day N
- Predict days N+1 to N+7
- Compare predictions to actuals
- Slide forward and repeat

This gives a realistic estimate of prediction accuracy over time.
"""

import os
import sys
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt

from config import (
    CSV_DIR, IMG_DIR, MODEL_DIR,
    COMBINED_CSV, MODEL_FILE, FEATURE_COLS_FILE,
    ensure_directories
)

# Hindcast output files
HINDCAST_CSV = f"{CSV_DIR}/hindcast_results.csv"
HINDCAST_METRICS_CSV = f"{CSV_DIR}/hindcast_metrics.csv"
HINDCAST_PLOT = f"{IMG_DIR}/hindcast_results.png"

# Default parameters
DEFAULT_FORECAST_HORIZON = 7  # Predict 7 days ahead
DEFAULT_STEP_SIZE = 7  # Move forward 7 days between evaluations
DEFAULT_MIN_TRAINING_DAYS = 60  # Minimum training data before starting hindcast


def create_features(df):
    """Create features for the model (same as train.py)."""
    df = df.copy()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofmonth'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = (df['dayofmonth'] <= 3).astype(int)
    df['is_month_end'] = (df['dayofmonth'] >= 28).astype(int)
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'lag_{lag}'] = df['dsa_count'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).max()
    
    # Trend features
    for window in [7, 14]:
        col_name = f'trend_{window}'
        df[col_name] = 0.0
        for i in range(window, len(df)):
            y = df['dsa_count'].iloc[i-window:i].values
            x = np.arange(window)
            if len(y) == window:
                coeffs = np.polyfit(x, y, 1)
                df.loc[df.index[i], col_name] = coeffs[0]
    
    # Day of week effect
    dow_means = df.groupby('dayofweek')['dsa_count'].transform('mean')
    overall_mean = df['dsa_count'].mean()
    df['dow_effect'] = dow_means / overall_mean if overall_mean > 0 else 1
    
    # Month effect
    month_means = df.groupby('month')['dsa_count'].transform('mean')
    df['month_effect'] = month_means / overall_mean if overall_mean > 0 else 1
    
    # Sentiment features
    if 'sentiment_mean' in df.columns:
        df['sentiment_lag_1'] = df['sentiment_mean'].shift(1)
        df['sentiment_lag_3'] = df['sentiment_mean'].shift(3)
        df['sentiment_lag_7'] = df['sentiment_mean'].shift(7)
        df['sentiment_rolling_7'] = df['sentiment_mean'].shift(1).rolling(window=7, min_periods=1).mean()
    
    if 'negative_news_pct' in df.columns:
        df['negative_news_rolling_7'] = df['negative_news_pct'].shift(1).rolling(window=7, min_periods=1).mean()
    
    return df


def get_feature_columns():
    """Get the list of feature columns to use."""
    return [
        'dayofweek', 'dayofmonth', 'month', 'year', 'weekofyear',
        'is_weekend', 'is_month_start', 'is_month_end',
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
        'rolling_mean_28', 'rolling_std_28', 'rolling_min_28', 'rolling_max_28',
        'trend_7', 'trend_14',
        'dow_effect', 'month_effect',
        'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_lag_7',
        'sentiment_rolling_7', 'negative_news_rolling_7'
    ]


def train_model_on_subset(df, feature_cols):
    """Train a model on a subset of data."""
    available_features = [col for col in feature_cols if col in df.columns]
    
    df_clean = df.dropna(subset=available_features + ['dsa_count']).copy()
    
    if len(df_clean) < 30:
        return None, available_features
    
    X = df_clean[available_features]
    y = df_clean['dsa_count']
    
    model = GradientBoostingRegressor(
        n_estimators=100,  # Fewer trees for faster training
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    return model, available_features


def predict_future(model, df, feature_cols, horizon=7):
    """Make predictions for the next N days after the last date in df."""
    last_date = df['date'].max()
    last_row = df.iloc[-1]
    recent_values = df['dsa_count'].tail(28).tolist()
    
    predictions = []
    
    for i in range(1, horizon + 1):
        future_date = last_date + timedelta(days=i)
        
        # Create features for future date
        row_features = {
            'date': future_date,
            'dayofweek': future_date.weekday(),
            'dayofmonth': future_date.day,
            'month': future_date.month,
            'year': future_date.year,
            'weekofyear': future_date.isocalendar()[1],
            'is_weekend': 1 if future_date.weekday() >= 5 else 0,
            'is_month_start': 1 if future_date.day <= 3 else 0,
            'is_month_end': 1 if future_date.day >= 28 else 0,
        }
        
        # Add lag features from recent values
        for lag in [1, 2, 3, 7, 14, 28]:
            col = f'lag_{lag}'
            if col in feature_cols:
                if lag <= len(recent_values):
                    row_features[col] = recent_values[-lag]
                else:
                    row_features[col] = np.mean(recent_values)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            if f'rolling_mean_{window}' in feature_cols:
                row_features[f'rolling_mean_{window}'] = np.mean(recent_values[-window:])
            if f'rolling_std_{window}' in feature_cols:
                row_features[f'rolling_std_{window}'] = np.std(recent_values[-window:])
            if f'rolling_min_{window}' in feature_cols:
                row_features[f'rolling_min_{window}'] = np.min(recent_values[-window:])
            if f'rolling_max_{window}' in feature_cols:
                row_features[f'rolling_max_{window}'] = np.max(recent_values[-window:])
        
        # Trend features
        for window in [7, 14]:
            col = f'trend_{window}'
            if col in feature_cols:
                if len(recent_values) >= window:
                    y = recent_values[-window:]
                    x = np.arange(window)
                    coeffs = np.polyfit(x, y, 1)
                    row_features[col] = coeffs[0]
                else:
                    row_features[col] = 0
        
        # Effect features from last known values
        for col in ['dow_effect', 'month_effect']:
            if col in feature_cols and col in df.columns:
                row_features[col] = last_row[col] if col in last_row else 1
        
        # Sentiment features
        for col in ['sentiment_lag_1', 'sentiment_lag_3', 'sentiment_lag_7', 
                    'sentiment_rolling_7', 'negative_news_rolling_7']:
            if col in feature_cols:
                if col in df.columns:
                    row_features[col] = df[col].iloc[-1] if not pd.isna(df[col].iloc[-1]) else 0
                else:
                    row_features[col] = 0
        
        # Make prediction
        X_pred = pd.DataFrame([row_features])[feature_cols]
        pred = model.predict(X_pred)[0]
        pred = max(0, pred)  # Ensure non-negative
        
        predictions.append({
            'target_date': future_date,
            'predicted': pred,
            'horizon': i
        })
    
    return predictions


def run_hindcast(df, forecast_horizon=DEFAULT_FORECAST_HORIZON, 
                 step_size=DEFAULT_STEP_SIZE, 
                 min_training_days=DEFAULT_MIN_TRAINING_DAYS,
                 retrain_every=None):
    """
    Run walk-forward hindcast validation.
    
    Args:
        df: DataFrame with historical data
        forecast_horizon: Number of days to predict ahead
        step_size: Days to move forward between evaluations
        min_training_days: Minimum training data before starting
        retrain_every: Retrain model every N steps (None = retrain every step)
    
    Returns:
        DataFrame with hindcast results
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    feature_cols = get_feature_columns()
    
    # Determine start and end dates for hindcast
    min_date = df['date'].min()
    max_date = df['date'].max()
    start_date = min_date + timedelta(days=min_training_days)
    end_date = max_date - timedelta(days=forecast_horizon)
    
    if start_date >= end_date:
        print("Not enough data for hindcast")
        return pd.DataFrame()
    
    print(f"  Hindcast period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print(f"  Step size: {step_size} days")
    
    results = []
    current_date = start_date
    step_count = 0
    model = None
    
    while current_date <= end_date:
        # Get training data (all data up to current_date)
        train_df = df[df['date'] <= current_date].copy()
        train_df = create_features(train_df)
        
        # Train or reuse model
        if model is None or retrain_every is None or step_count % retrain_every == 0:
            model, available_features = train_model_on_subset(train_df, feature_cols)
            if model is None:
                current_date += timedelta(days=step_size)
                step_count += 1
                continue
        
        # Make predictions
        predictions = predict_future(model, train_df, available_features, forecast_horizon)
        
        # Get actuals and compare
        for pred in predictions:
            target_date = pred['target_date']
            actual_row = df[df['date'] == target_date]
            
            if len(actual_row) > 0:
                actual = actual_row['dsa_count'].values[0]
                error = actual - pred['predicted']
                abs_error = abs(error)
                pct_error = (abs_error / actual * 100) if actual > 0 else 0
                
                results.append({
                    'prediction_date': current_date,
                    'target_date': target_date,
                    'horizon': pred['horizon'],
                    'predicted': pred['predicted'],
                    'actual': actual,
                    'error': error,
                    'abs_error': abs_error,
                    'pct_error': pct_error
                })
        
        current_date += timedelta(days=step_size)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"  Processed {step_count} steps...")
    
    return pd.DataFrame(results)


def calculate_metrics(results_df):
    """Calculate summary metrics from hindcast results."""
    if results_df.empty:
        return pd.DataFrame()
    
    metrics = []
    
    # Overall metrics
    overall = {
        'horizon': 'Overall',
        'count': len(results_df),
        'mae': results_df['abs_error'].mean(),
        'rmse': np.sqrt((results_df['error'] ** 2).mean()),
        'mape': results_df['pct_error'].mean(),
        'r2': r2_score(results_df['actual'], results_df['predicted']),
        'median_ape': results_df['pct_error'].median(),
        'pct_under_10': (results_df['pct_error'] < 10).mean() * 100,
        'pct_under_20': (results_df['pct_error'] < 20).mean() * 100,
        'pct_under_50': (results_df['pct_error'] < 50).mean() * 100,
    }
    metrics.append(overall)
    
    # Metrics by horizon
    for horizon in sorted(results_df['horizon'].unique()):
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        horizon_metrics = {
            'horizon': f'Day {horizon}',
            'count': len(horizon_df),
            'mae': horizon_df['abs_error'].mean(),
            'rmse': np.sqrt((horizon_df['error'] ** 2).mean()),
            'mape': horizon_df['pct_error'].mean(),
            'r2': r2_score(horizon_df['actual'], horizon_df['predicted']),
            'median_ape': horizon_df['pct_error'].median(),
            'pct_under_10': (horizon_df['pct_error'] < 10).mean() * 100,
            'pct_under_20': (horizon_df['pct_error'] < 20).mean() * 100,
            'pct_under_50': (horizon_df['pct_error'] < 50).mean() * 100,
        }
        metrics.append(horizon_metrics)
    
    return pd.DataFrame(metrics)


def plot_hindcast_results(results_df, metrics_df, filename=HINDCAST_PLOT):
    """Create visualization of hindcast results."""
    print(f"\nCreating hindcast visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Predicted scatter
    ax1 = axes[0, 0]
    ax1.scatter(results_df['actual'], results_df['predicted'], 
                alpha=0.3, s=20, c='steelblue', edgecolors='none')
    
    min_val = min(results_df['actual'].min(), results_df['predicted'].min())
    max_val = max(results_df['actual'].max(), results_df['predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    r2 = r2_score(results_df['actual'], results_df['predicted'])
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Hindcast: Actual vs Predicted (R² = {r2:.3f})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAPE by horizon
    ax2 = axes[0, 1]
    horizon_mape = results_df.groupby('horizon')['pct_error'].mean()
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(horizon_mape)))
    bars = ax2.bar(horizon_mape.index, horizon_mape.values, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Forecast Horizon (Days)')
    ax2.set_ylabel('Mean Absolute Percentage Error (%)')
    ax2.set_title('Prediction Accuracy by Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, horizon_mape.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Error distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['pct_error'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=results_df['pct_error'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["pct_error"].mean():.1f}%')
    ax3.axvline(x=results_df['pct_error'].median(), color='orange', linestyle='--', 
                linewidth=2, label=f'Median: {results_df["pct_error"].median():.1f}%')
    ax3.set_xlabel('Percentage Error (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of predictions vs actuals (sample)
    ax4 = axes[1, 1]
    
    # Get day-1 predictions only for cleaner visualization
    day1 = results_df[results_df['horizon'] == 1].copy()
    day1 = day1.sort_values('target_date').tail(60)  # Last 60 points
    
    ax4.plot(day1['target_date'], day1['actual'], 'b-', linewidth=1.5, label='Actual', marker='o', markersize=3)
    ax4.plot(day1['target_date'], day1['predicted'], 'r--', linewidth=1.5, label='Predicted (Day+1)', marker='s', markersize=3)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('DSA Count')
    ax4.set_title('Day-1 Predictions Over Time (Last 60)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("DSA HINDCAST (BACKTESTING)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    ensure_directories()
    
    if not os.path.exists(COMBINED_CSV):
        print(f"\nError: {COMBINED_CSV} not found. Run `make backfill` first.")
        return
    
    print(f"\nLoading historical data...")
    df = pd.read_csv(COMBINED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nRunning hindcast...")
    results_df = run_hindcast(
        df,
        forecast_horizon=7,
        step_size=7,
        min_training_days=60,
        retrain_every=4  # Retrain every 4 weeks
    )
    
    if results_df.empty:
        print("No hindcast results generated.")
        return
    
    print(f"\n  Generated {len(results_df)} predictions")
    
    # Save results
    results_df.to_csv(HINDCAST_CSV, index=False)
    print(f"  Saved results to {HINDCAST_CSV}")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics_df = calculate_metrics(results_df)
    metrics_df.to_csv(HINDCAST_METRICS_CSV, index=False)
    print(f"  Saved metrics to {HINDCAST_METRICS_CSV}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("HINDCAST RESULTS")
    print("=" * 60)
    
    overall = metrics_df[metrics_df['horizon'] == 'Overall'].iloc[0]
    print(f"\nOverall Performance ({int(overall['count'])} predictions):")
    print(f"  MAE:  {overall['mae']:,.0f}")
    print(f"  RMSE: {overall['rmse']:,.0f}")
    print(f"  MAPE: {overall['mape']:.1f}%")
    print(f"  R²:   {overall['r2']:.4f}")
    print(f"\n  % within 10% error: {overall['pct_under_10']:.1f}%")
    print(f"  % within 20% error: {overall['pct_under_20']:.1f}%")
    print(f"  % within 50% error: {overall['pct_under_50']:.1f}%")
    
    print("\nPerformance by Horizon:")
    print("-" * 50)
    for _, row in metrics_df[metrics_df['horizon'] != 'Overall'].iterrows():
        print(f"  {row['horizon']}: MAPE={row['mape']:.1f}%, R²={row['r2']:.3f}")
    
    # Create visualization
    plot_hindcast_results(results_df, metrics_df)
    
    print("\n" + "=" * 60)
    print("HINDCAST COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  {HINDCAST_CSV}")
    print(f"  {HINDCAST_METRICS_CSV}")
    print(f"  {HINDCAST_PLOT}")


if __name__ == "__main__":
    main()

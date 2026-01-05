#!/usr/bin/env python3

import os
import sys
import subprocess
import json
from datetime import datetime, timedelta

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q", "--break-system-packages"])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
import joblib

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import matplotlib.dates as mdates
except:
    pass

import warnings
warnings.filterwarnings('ignore')

from config import (
    CSV_DIR, IMG_DIR,
    COMBINED_CSV, MODEL_FILE, FEATURE_COLS_FILE,
    PREDICTIONS_CSV, ACCURACY_PLOT, FORECAST_PLOT,
    ensure_directories
)

FORECAST_DAYS = 7


def create_features(df):
    df = df.copy()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofmonth'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = (df['dayofmonth'] <= 3).astype(int)
    df['is_month_end'] = (df['dayofmonth'] >= 28).astype(int)
    
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'lag_{lag}'] = df['dsa_count'].shift(lag)
    
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df['dsa_count'].shift(1).rolling(window=window, min_periods=1).max()
    
    for window in [7, 14]:
        col_name = f'trend_{window}'
        df[col_name] = 0.0
        for i in range(window, len(df)):
            y = df['dsa_count'].iloc[i-window:i].values
            x = np.arange(window)
            if len(y) == window:
                coeffs = np.polyfit(x, y, 1)
                df.loc[df.index[i], col_name] = coeffs[0]
    
    dow_means = df.groupby('dayofweek')['dsa_count'].transform('mean')
    overall_mean = df['dsa_count'].mean()
    df['dow_effect'] = dow_means / overall_mean if overall_mean > 0 else 1
    
    month_means = df.groupby('month')['dsa_count'].transform('mean')
    df['month_effect'] = month_means / overall_mean if overall_mean > 0 else 1
    
    if 'sentiment_mean' in df.columns:
        df['sentiment_lag_1'] = df['sentiment_mean'].shift(1)
        df['sentiment_lag_3'] = df['sentiment_mean'].shift(3)
        df['sentiment_lag_7'] = df['sentiment_mean'].shift(7)
        df['sentiment_rolling_7'] = df['sentiment_mean'].shift(1).rolling(window=7, min_periods=1).mean()
    
    if 'negative_news_pct' in df.columns:
        df['negative_news_rolling_7'] = df['negative_news_pct'].shift(1).rolling(window=7, min_periods=1).mean()
    
    return df


def create_future_dates(df, days=7):
    last_date = df['date'].max()
    last_row = df.iloc[-1].copy()
    
    future_dates = []
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        future_dates.append({
            'date': future_date,
            'dayofweek': future_date.weekday(),
            'dayofmonth': future_date.day,
            'month': future_date.month,
            'year': future_date.year,
            'weekofyear': future_date.isocalendar()[1],
            'is_weekend': 1 if future_date.weekday() >= 5 else 0,
            'is_month_start': 1 if future_date.day <= 3 else 0,
            'is_month_end': 1 if future_date.day >= 28 else 0
        })
    
    future_df = pd.DataFrame(future_dates)
    return future_df, last_row, df


def prepare_future_features(future_df, last_row, historical_df, feature_cols):
    prepared = future_df.copy()
    
    recent_values = historical_df['dsa_count'].tail(28).tolist()
    
    for col in feature_cols:
        if col not in prepared.columns:
            if col.startswith('lag_'):
                lag = int(col.split('_')[1])
                if lag <= len(recent_values):
                    prepared[col] = recent_values[-lag]
                else:
                    prepared[col] = np.mean(recent_values)
            elif col.startswith('rolling_'):
                parts = col.split('_')
                window = int(parts[-1])
                if 'mean' in col:
                    prepared[col] = np.mean(recent_values[-window:])
                elif 'std' in col:
                    prepared[col] = np.std(recent_values[-window:])
                elif 'min' in col:
                    prepared[col] = np.min(recent_values[-window:])
                elif 'max' in col:
                    prepared[col] = np.max(recent_values[-window:])
            elif col.startswith('trend_'):
                window = int(col.split('_')[1])
                if len(recent_values) >= window:
                    y = recent_values[-window:]
                    x = np.arange(window)
                    coeffs = np.polyfit(x, y, 1)
                    prepared[col] = coeffs[0]
                else:
                    prepared[col] = 0
            elif col == 'dow_effect':
                dow_effects = historical_df.groupby('dayofweek')['dsa_count'].mean()
                overall_mean = historical_df['dsa_count'].mean()
                prepared[col] = prepared['dayofweek'].map(lambda x: dow_effects.get(x, overall_mean) / overall_mean)
            elif col == 'month_effect':
                month_effects = historical_df.groupby('month')['dsa_count'].mean()
                overall_mean = historical_df['dsa_count'].mean()
                prepared[col] = prepared['month'].map(lambda x: month_effects.get(x, overall_mean) / overall_mean)
            elif col.startswith('sentiment'):
                if col in historical_df.columns:
                    prepared[col] = historical_df[col].iloc[-1] if not pd.isna(historical_df[col].iloc[-1]) else 0
                else:
                    prepared[col] = 0
            elif col.startswith('negative_news'):
                if col in historical_df.columns:
                    prepared[col] = historical_df[col].iloc[-1] if not pd.isna(historical_df[col].iloc[-1]) else 0
                else:
                    prepared[col] = 0
            else:
                if col in last_row.index:
                    prepared[col] = last_row[col]
                else:
                    prepared[col] = 0
    
    return prepared


def make_predictions(model, prepared_df, feature_cols):
    X = prepared_df[feature_cols].values
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)
    return predictions


def save_predictions(predictions_df):
    predictions_df['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predictions_df['target_date'] = predictions_df['target_date'].dt.strftime('%Y-%m-%d')
    predictions_df['actual'] = None
    predictions_df['error'] = None
    predictions_df['abs_error'] = None
    predictions_df['pct_error'] = None
    
    if os.path.exists(PREDICTIONS_CSV):
        existing = pd.read_csv(PREDICTIONS_CSV)
        combined = pd.concat([existing, predictions_df], ignore_index=True)
        combined.to_csv(PREDICTIONS_CSV, index=False)
    else:
        predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    
    print(f"  Saved predictions to {PREDICTIONS_CSV}")


def update_with_actuals():
    if not os.path.exists(PREDICTIONS_CSV) or not os.path.exists(COMBINED_CSV):
        return pd.DataFrame()
    
    predictions = pd.read_csv(PREDICTIONS_CSV)
    actuals = pd.read_csv(COMBINED_CSV)
    
    actuals['date'] = pd.to_datetime(actuals['date']).dt.strftime('%Y-%m-%d')
    actual_dict = dict(zip(actuals['date'], actuals['dsa_count']))
    
    updated = False
    for idx, row in predictions.iterrows():
        target_date = row['target_date']
        if target_date in actual_dict and pd.isna(row['actual']):
            actual_value = actual_dict[target_date]
            predictions.loc[idx, 'actual'] = actual_value
            predictions.loc[idx, 'error'] = actual_value - row['predicted']
            predictions.loc[idx, 'abs_error'] = abs(actual_value - row['predicted'])
            if actual_value > 0:
                predictions.loc[idx, 'pct_error'] = abs(actual_value - row['predicted']) / actual_value * 100
            updated = True
    
    if updated:
        predictions.to_csv(PREDICTIONS_CSV, index=False)
        print(f"  Updated predictions with actuals")
    
    return predictions


def plot_forecast(historical_df, forecast_df, filename=FORECAST_PLOT):
    print(f"\nCreating forecast visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    recent_days = 30
    recent_df = historical_df.tail(recent_days)
    
    ax.plot(recent_df['date'], recent_df['dsa_count'], 'b-', linewidth=2, label='Historical', marker='o', markersize=4)
    
    forecast_dates = forecast_df['target_date']
    forecast_values = forecast_df['predicted']
    
    last_hist_date = recent_df['date'].iloc[-1]
    last_hist_value = recent_df['dsa_count'].iloc[-1]
    
    connection_dates = pd.concat([pd.Series([last_hist_date]), forecast_dates])
    connection_values = pd.concat([pd.Series([last_hist_value]), forecast_values])
    
    ax.plot(connection_dates, connection_values, 'r--', linewidth=2, label='Forecast', marker='s', markersize=6)
    
    ax.fill_between(forecast_dates, forecast_values * 0.9, forecast_values * 1.1, alpha=0.2, color='red', label='10% confidence')
    
    ax.axvline(x=last_hist_date, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('DSA Violation Count', fontsize=11)
    ax.set_title('DSA Violations: Historical + 7-Day Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_accuracy(df, filename=ACCURACY_PLOT):
    print(f"\nCreating accuracy visualization...")
    
    df_with_actuals = df[df['actual'].notna()].copy()
    
    if len(df_with_actuals) == 0:
        print(f"  No predictions with actuals available yet")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No predictions with actuals available yet.\nRun daily to collect prediction accuracy data.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()
        return
    
    df_with_actuals['target_date'] = pd.to_datetime(df_with_actuals['target_date'])
    df_with_actuals['abs_error'] = df_with_actuals.apply(
        lambda x: abs(x['actual'] - x['predicted']) if x['actual'] is not None else None, axis=1
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(df_with_actuals['target_date'], df_with_actuals['actual'], 'b-', label='Actual', linewidth=2, marker='o')
    ax1.plot(df_with_actuals['target_date'], df_with_actuals['predicted'], 'r--', label='Predicted', linewidth=2, marker='s')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('DSA Count')
    ax1.set_title('Predicted vs Actual Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    ax2.scatter(df_with_actuals['actual'], df_with_actuals['predicted'], alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(df_with_actuals['actual'].min(), df_with_actuals['predicted'].min())
    max_val = max(df_with_actuals['actual'].max(), df_with_actuals['predicted'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    from sklearn.metrics import r2_score
    r2 = r2_score(df_with_actuals['actual'], df_with_actuals['predicted'])
    
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Scatter Plot (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    residuals = df_with_actuals['actual'] - df_with_actuals['predicted']
    ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual (Actual - Predicted)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    valid_pct = df_with_actuals[df_with_actuals['pct_error'].notna()]
    if len(valid_pct) > 0:
        colors = ['green' if e < 20 else 'orange' if e < 50 else 'red' for e in valid_pct['pct_error']]
        ax4.bar(valid_pct['target_date'], valid_pct['pct_error'], color=colors, alpha=0.7)
        ax4.axhline(y=20, color='green', linestyle='--', linewidth=1, label='20% threshold')
        ax4.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='50% threshold')
        
        avg_mape = valid_pct['pct_error'].mean()
        ax4.axhline(y=avg_mape, color='blue', linestyle='-', linewidth=2, label=f'Avg MAPE: {avg_mape:.1f}%')
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Percentage Error (%)')
        ax4.set_title('Percentage Error by Prediction', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No percentage error data available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    if len(valid_pct) > 0:
        print(f"\nAccuracy Summary:")
        print(f"  Predictions evaluated: {len(valid_pct)}")
        print(f"  Mean Absolute Error: {df_with_actuals['abs_error'].mean():,.0f}")
        print(f"  Mean Percentage Error: {valid_pct['pct_error'].mean():.1f}%")
        print(f"  R²: {r2:.4f}")


def main():
    print("=" * 60)
    print("DSA INFERENCE & FORECASTING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    ensure_directories()
    
    if not os.path.exists(MODEL_FILE):
        print(f"\nError: {MODEL_FILE} not found. Run train_model.py first.")
        return
    
    if not os.path.exists(FEATURE_COLS_FILE):
        print(f"\nError: {FEATURE_COLS_FILE} not found. Run train_model.py first.")
        return
    
    print(f"\nLoading model...")
    model = joblib.load(MODEL_FILE)
    
    with open(FEATURE_COLS_FILE, 'r') as f:
        feature_cols = json.load(f)
    print(f"  Model loaded with {len(feature_cols)} features")
    
    if not os.path.exists(COMBINED_CSV):
        print(f"\nError: {COMBINED_CSV} not found. Run backfill_data.py first.")
        return
    
    print(f"\nLoading historical data...")
    df = pd.read_csv(COMBINED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} records")
    print(f"  Latest date: {df['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nPreparing features...")
    df = create_features(df)
    
    print(f"\nGenerating {FORECAST_DAYS}-day forecast...")
    future_df, last_row, historical = create_future_dates(df, days=FORECAST_DAYS)
    prepared_future = prepare_future_features(future_df, last_row, df, feature_cols)
    
    predictions = make_predictions(model, prepared_future, feature_cols)
    
    forecast_df = pd.DataFrame({
        'target_date': future_df['date'],
        'predicted': predictions
    })
    
    print(f"\nForecast for next {FORECAST_DAYS} days:")
    print("-" * 40)
    for _, row in forecast_df.iterrows():
        date_str = row['target_date'].strftime('%Y-%m-%d (%A)')
        print(f"  {date_str}: {row['predicted']:,.0f}")
    print("-" * 40)
    print(f"  Average: {forecast_df['predicted'].mean():,.0f}")
    print(f"  Total:   {forecast_df['predicted'].sum():,.0f}")
    
    print(f"\nSaving predictions...")
    save_predictions(forecast_df.copy())
    
    print(f"\nUpdating accuracy tracking...")
    predictions_with_actuals = update_with_actuals()
    
    plot_forecast(df, forecast_df)
    plot_accuracy(predictions_with_actuals)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"\nFiles created/updated:")
    print(f"  {PREDICTIONS_CSV}")
    print(f"  {FORECAST_PLOT}")
    print(f"  {ACCURACY_PLOT}")


if __name__ == "__main__":
    main()
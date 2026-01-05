#!/usr/bin/env python3

import os
import sys
import subprocess
import json
from datetime import datetime

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

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    install("scikit-learn")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    CSV_DIR, IMG_DIR,
    COMBINED_CSV, MODEL_FILE, FEATURE_COLS_FILE,
    TRAINING_PLOT, FEATURE_IMPORTANCE_PLOT,
    ensure_directories
)


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


def train_model(df, target_col='dsa_count', validation_days=30):
    feature_cols = [
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
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    df_model = df.dropna(subset=available_features + [target_col]).copy()
    
    print(f"  Samples after dropping NaN: {len(df_model)}")
    
    split_idx = len(df_model) - validation_days
    train_df = df_model.iloc[:split_idx]
    val_df = df_model.iloc[split_idx:]
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    
    X_train = train_df[available_features]
    y_train = train_df[target_col]
    X_val = val_df[available_features]
    y_val = val_df[target_col]
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_metrics = {
        'mae': mean_absolute_error(y_train, train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'r2': r2_score(y_train, train_pred),
        'mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100
    }
    
    val_metrics = {
        'mae': mean_absolute_error(y_val, val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'r2': r2_score(y_val, val_pred),
        'mape': np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    }
    
    return model, available_features, train_metrics, val_metrics, val_df, val_pred


def plot_training_results(val_df, val_pred, train_metrics, val_metrics, filename=TRAINING_PLOT):
    print(f"\nCreating training visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(val_df['date'], val_df['dsa_count'], 'b-', label='Actual', linewidth=2)
    ax1.plot(val_df['date'], val_pred, 'r--', label='Predicted', linewidth=2)
    ax1.fill_between(val_df['date'], val_df['dsa_count'], val_pred, alpha=0.3, color='gray')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('DSA Count')
    ax1.set_title('Validation: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    ax2.scatter(val_df['dsa_count'], val_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(val_df['dsa_count'].min(), min(val_pred))
    max_val = max(val_df['dsa_count'].max(), max(val_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual DSA Count')
    ax2.set_ylabel('Predicted DSA Count')
    ax2.set_title(f'Scatter Plot (R² = {val_metrics["r2"]:.4f})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    residuals = val_df['dsa_count'].values - val_pred
    ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual (Actual - Predicted)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    metrics_text = f"""Training Metrics:
  MAE:  {train_metrics['mae']:,.0f}
  RMSE: {train_metrics['rmse']:,.0f}
  R²:   {train_metrics['r2']:.4f}
  MAPE: {train_metrics['mape']:.2f}%

Validation Metrics:
  MAE:  {val_metrics['mae']:,.0f}
  RMSE: {val_metrics['rmse']:,.0f}
  R²:   {val_metrics['r2']:.4f}
  MAPE: {val_metrics['mape']:.2f}%"""
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.axis('off')
    ax4.set_title('Model Performance Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_feature_importance(model, feature_cols, filename=FEATURE_IMPORTANCE_PLOT):
    print(f"\nCreating feature importance plot...")
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    top_n = min(20, len(feature_cols))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importance[top_indices][::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[i] for i in top_indices][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(importance[top_indices][::-1]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("DSA MODEL TRAINING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    ensure_directories()
    
    if not os.path.exists(COMBINED_CSV):
        print(f"\nError: {COMBINED_CSV} not found. Run backfill_data.py first.")
        return
    
    print(f"\nLoading data from {COMBINED_CSV}...")
    df = pd.read_csv(COMBINED_CSV)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nCreating features...")
    df = create_features(df)
    print(f"  Created {len([c for c in df.columns if c not in ['date', 'dsa_count']])} features")
    
    print(f"\nTraining model...")
    model, feature_cols, train_metrics, val_metrics, val_df, val_pred = train_model(df)
    
    print(f"\nTraining Metrics:")
    print(f"  MAE:  {train_metrics['mae']:,.0f}")
    print(f"  RMSE: {train_metrics['rmse']:,.0f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    print(f"  MAPE: {train_metrics['mape']:.2f}%")
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {val_metrics['mae']:,.0f}")
    print(f"  RMSE: {val_metrics['rmse']:,.0f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"  MAPE: {val_metrics['mape']:.2f}%")
    
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved: {MODEL_FILE}")
    
    with open(FEATURE_COLS_FILE, 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved: {FEATURE_COLS_FILE}")
    
    plot_training_results(val_df, val_pred, train_metrics, val_metrics)
    plot_feature_importance(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  {MODEL_FILE}")
    print(f"  {FEATURE_COLS_FILE}")
    print(f"  {TRAINING_PLOT}")
    print(f"  {FEATURE_IMPORTANCE_PLOT}")
    
    print(f"\nValidation Performance:")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"  MAE:  {val_metrics['mae']:,.0f}")
    print(f"  MAPE: {val_metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
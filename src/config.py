#!/usr/bin/env python3

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEWS_API_TOKEN = os.getenv("NEWS_API_TOKEN", "")
NEWS_API_PLAN = os.getenv("NEWS_API_PLAN", "free").lower()

PLANS = {
    "free": {"articles_per_request": 3, "daily_requests": 100},
    "basic": {"articles_per_request": 25, "daily_requests": 2500}
}

if NEWS_API_PLAN not in PLANS:
    NEWS_API_PLAN = "free"

NEWS_ARTICLES_PER_REQUEST = PLANS[NEWS_API_PLAN]["articles_per_request"]
NEWS_DAILY_REQUESTS = PLANS[NEWS_API_PLAN]["daily_requests"]

NEWS_API_TOP = "https://api.thenewsapi.com/v1/news/top"
NEWS_API_ALL = "https://api.thenewsapi.com/v1/news/all"

CSV_DIR = "csv"
IMG_DIR = "img"
MODEL_DIR = "model"

DSA_CSV = f"{CSV_DIR}/dsa_violations.csv"
NEWS_CSV = f"{CSV_DIR}/news_history.csv"
COMBINED_CSV = f"{CSV_DIR}/dsa_news_combined.csv"
PREDICTIONS_CSV = f"{CSV_DIR}/predictions_history.csv"

MODEL_FILE = f"{MODEL_DIR}/dsa_model.pkl"
FEATURE_COLS_FILE = f"{MODEL_DIR}/feature_cols.json"

BACKFILL_PLOT = f"{IMG_DIR}/backfill_visualization.png"
TRAINING_PLOT = f"{IMG_DIR}/training_results.png"
FEATURE_IMPORTANCE_PLOT = f"{IMG_DIR}/feature_importance.png"
FORECAST_PLOT = f"{IMG_DIR}/forecast_next_week.png"
ACCURACY_PLOT = f"{IMG_DIR}/prediction_accuracy.png"


class RateLimitError(Exception):
    pass


class UsageLimitError(Exception):
    pass


class ConfigurationError(Exception):
    pass


def validate_config():
    errors = []
    warnings = []
    
    if not NEWS_API_TOKEN:
        errors.append("NEWS_API_TOKEN is not set")
    elif NEWS_API_TOKEN == "your_api_token_here":
        errors.append("NEWS_API_TOKEN is still set to placeholder")
    
    return errors, warnings


def print_config():
    if len(NEWS_API_TOKEN) >= 4:
        token_display = f"{'*' * 8}...{NEWS_API_TOKEN[-4:]}"
    elif NEWS_API_TOKEN:
        token_display = "****"
    else:
        token_display = "(not set)"
    print(f"\nConfiguration:")
    print(f"  Plan: {NEWS_API_PLAN}")
    print(f"  API Token: {token_display}")
    print(f"  Articles per day: {NEWS_ARTICLES_PER_REQUEST}")
    print(f"  Daily request limit: {NEWS_DAILY_REQUESTS}")


def ensure_directories():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
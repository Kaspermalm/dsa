#!/usr/bin/env python3

import os
import sys
import subprocess
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import time

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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    install("vaderSentiment")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    install("python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from config import (
    NEWS_API_TOKEN, NEWS_ARTICLES_PER_REQUEST, NEWS_DAILY_REQUESTS, NEWS_API_PLAN,
    NEWS_API_TOP, NEWS_API_ALL,
    CSV_DIR, IMG_DIR,
    DSA_CSV, NEWS_CSV, COMBINED_CSV, BACKFILL_PLOT,
    RateLimitError, UsageLimitError,
    validate_config, print_config, ensure_directories
)

POWERBI_URL = "https://wabi-north-europe-k-primary-api.analysis.windows.net/public/reports/querydata?synchronous=true"
POWERBI_RESOURCE_KEY = "c7c7f08e-3968-4a5e-968f-1a70a6692770"
POWERBI_DATASET_ID = "87005e9b-9279-4f27-b0c2-259abfc79dbc"
POWERBI_REPORT_ID = "592df1f0-7547-46cf-8877-185c7bc9b3a3"
POWERBI_MODEL_ID = 2529495

TARGET_VIOLATIONS = [
    "Risk for public security",
    "Negative effects on civic discourse or elections",
    "Illegal or harmful speech"
]


def fetch_dsa_data(days_back=365):
    days_back = min(days_back, 365)
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    headers = {
        "Content-Type": "application/json",
        "X-PowerBI-ResourceKey": POWERBI_RESOURCE_KEY
    }
    
    payload = {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "a", "Entity": "main", "Type": 0},
                                {"Name": "v", "Entity": "ViolationCounts", "Type": 0},
                                {"Name": "p", "Entity": "platforms", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "HierarchyLevel": {
                                        "Expression": {
                                            "Hierarchy": {
                                                "Expression": {
                                                    "PropertyVariationSource": {
                                                        "Expression": {"SourceRef": {"Source": "a"}},
                                                        "Name": "Variation",
                                                        "Property": "Month"
                                                    }
                                                },
                                                "Hierarchy": "Date Hierarchy"
                                            }
                                        },
                                        "Level": "Year"
                                    },
                                    "Name": "Year"
                                },
                                {
                                    "HierarchyLevel": {
                                        "Expression": {
                                            "Hierarchy": {
                                                "Expression": {
                                                    "PropertyVariationSource": {
                                                        "Expression": {"SourceRef": {"Source": "a"}},
                                                        "Name": "Variation",
                                                        "Property": "Month"
                                                    }
                                                },
                                                "Hierarchy": "Date Hierarchy"
                                            }
                                        },
                                        "Level": "Month"
                                    },
                                    "Name": "Month"
                                },
                                {
                                    "HierarchyLevel": {
                                        "Expression": {
                                            "Hierarchy": {
                                                "Expression": {
                                                    "PropertyVariationSource": {
                                                        "Expression": {"SourceRef": {"Source": "a"}},
                                                        "Name": "Variation",
                                                        "Property": "Month"
                                                    }
                                                },
                                                "Hierarchy": "Date Hierarchy"
                                            }
                                        },
                                        "Level": "Day"
                                    },
                                    "Name": "Day"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "a"}},
                                                "Property": "total"
                                            }
                                        },
                                        "Function": 0
                                    },
                                    "Name": "Total"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "v"}},
                                        "Property": "Violation"
                                    },
                                    "Name": "Violation"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "Comparison": {
                                            "ComparisonKind": 2,
                                            "Left": {
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "a"}},
                                                    "Property": "Month"
                                                }
                                            },
                                            "Right": {
                                                "Literal": {"Value": f"datetime'{start_date.strftime('%Y-%m-%d')}T00:00:00'"}
                                            }
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "Not": {
                                            "Expression": {
                                                "In": {
                                                    "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "p"}}, "Property": "name"}}],
                                                    "Values": [[{"Literal": {"Value": "'DSA Team'"}}]]
                                                }
                                            }
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [
                                {"Direction": 1, "Expression": {"HierarchyLevel": {"Expression": {"Hierarchy": {"Expression": {"PropertyVariationSource": {"Expression": {"SourceRef": {"Source": "a"}}, "Name": "Variation", "Property": "Month"}}, "Hierarchy": "Date Hierarchy"}}, "Level": "Year"}}},
                                {"Direction": 1, "Expression": {"HierarchyLevel": {"Expression": {"Hierarchy": {"Expression": {"PropertyVariationSource": {"Expression": {"SourceRef": {"Source": "a"}}, "Name": "Variation", "Property": "Month"}}, "Hierarchy": "Date Hierarchy"}}, "Level": "Month"}}},
                                {"Direction": 1, "Expression": {"HierarchyLevel": {"Expression": {"Hierarchy": {"Expression": {"PropertyVariationSource": {"Expression": {"SourceRef": {"Source": "a"}}, "Name": "Variation", "Property": "Month"}}, "Hierarchy": "Date Hierarchy"}}, "Level": "Day"}}}
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2, 3]}]},
                            "Secondary": {"Groupings": [{"Projections": [4]}]},
                            "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 30000}}, "Secondary": {"Top": {"Count": 60}}},
                            "Version": 1
                        }
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": POWERBI_DATASET_ID,
                "Sources": [{"ReportId": POWERBI_REPORT_ID, "VisualId": ""}]
            }
        }],
        "modelId": POWERBI_MODEL_ID
    }
    
    try:
        response = requests.post(POWERBI_URL, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        rows = []
        results = data.get("results", [])
        if not results:
            print("  No results returned from API")
            return rows
        
        result = results[0].get("result", {})
        dsr = result.get("data", {}).get("dsr", {})
        data_shapes = dsr.get("DS", [])
        
        if not data_shapes:
            print("  No data shapes found")
            return rows
        
        ds = data_shapes[0]
        value_dicts = ds.get("ValueDicts", {})
        violations = value_dicts.get("D0", [])
        
        ph = ds.get("PH", [])
        if not ph:
            print("  No PH data found")
            return rows
        
        dm0 = ph[0].get("DM0", [])
        
        current_year = None
        current_month = None
        current_day = None
        
        for row in dm0:
            c = row.get("C", [])
            r = row.get("R", 0)
            
            if r == 0 and len(c) >= 3:
                current_year = c[0]
                current_month = c[1] + 1
                current_day = c[2]
            elif r == 3 and len(c) >= 1:
                current_day = c[0]
            elif len(c) == 2:
                current_month = c[0] + 1
                current_day = c[1]
            elif len(c) == 1:
                current_day = c[0]
            
            if current_year is None or current_month is None or current_day is None:
                continue
            
            date_str = f"{current_year}-{current_month:02d}-{current_day:02d}"
            
            x_data = row.get("X", [])
            if not x_data:
                continue
            
            violation_idx = -1
            for x_item in x_data:
                if "S" in x_item:
                    continue
                
                if "I" in x_item:
                    violation_idx = x_item["I"]
                else:
                    violation_idx += 1
                
                count = x_item.get("M0", 0)
                if count and count > 0 and violation_idx < len(violations):
                    violation = violations[violation_idx]
                    if violation in TARGET_VIOLATIONS:
                        rows.append({
                            "date": date_str,
                            "violation": violation,
                            "count": int(count)
                        })
        
        print(f"  Fetched {len(rows)} DSA rows")
        return rows
    
    except Exception as e:
        print(f"  Error fetching DSA data: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_news_for_date(date_str, limit=None):
    if limit is None:
        limit = NEWS_ARTICLES_PER_REQUEST
    
    params = {
        "api_token": NEWS_API_TOKEN,
        "locale": "us",
        "language": "en",
        "published_on": date_str,
        "limit": limit
    }
    
    try:
        response = requests.get(NEWS_API_TOP, params=params, timeout=30)
        
        if response.status_code == 429:
            raise RateLimitError("Rate limit reached (HTTP 429)")
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        if "error" in data:
            error_code = data["error"].get("code", "")
            error_msg = data["error"].get("message", str(data["error"]))
            
            if error_code == "rate_limit_reached":
                raise RateLimitError(f"Rate limit reached: {error_msg}")
            elif error_code == "usage_limit_reached":
                raise UsageLimitError(f"Daily usage limit reached: {error_msg}")
            else:
                return []
        
        articles = []
        if "data" in data:
            for article in data["data"]:
                articles.append({
                    "uuid": article.get("uuid", ""),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "snippet": article.get("snippet", ""),
                    "source": article.get("source", ""),
                    "categories": ",".join(article.get("categories", [])),
                    "published_at": article.get("published_at", "")
                })
        return articles
    except (RateLimitError, UsageLimitError):
        raise
    except Exception as e:
        return []


def calculate_sentiment(text, analyzer):
    if not text or not isinstance(text, str):
        return {"compound": 0, "neg": 0, "neu": 0, "pos": 0}
    return analyzer.polarity_scores(text)


def fetch_news_history(days_back=365):
    days_back = min(days_back, 365)
    max_requests = NEWS_DAILY_REQUESTS - 1
    
    print(f"\nFetching news headlines...")
    print(f"  Plan: {NEWS_API_PLAN} ({NEWS_ARTICLES_PER_REQUEST} articles/day)")
    print(f"  Target: {days_back} days (yesterday and older)")
    print(f"  Available requests: {max_requests}")
    
    existing_dates = set()
    if os.path.exists(NEWS_CSV):
        existing_df = pd.read_csv(NEWS_CSV)
        existing_dates = set(existing_df['date'].unique())
        print(f"  Already fetched: {len(existing_dates)} days")
    
    analyzer = SentimentIntensityAnalyzer()
    all_news = []
    
    current_date = datetime.now() - timedelta(days=1)
    end_date = current_date - timedelta(days=days_back)
    requests_made = 0
    days_fetched = 0
    
    try:
        while current_date >= end_date and requests_made < max_requests:
            date_str = current_date.strftime("%Y-%m-%d")
            
            if date_str in existing_dates:
                current_date -= timedelta(days=1)
                continue
            
            articles = fetch_news_for_date(date_str, limit=NEWS_ARTICLES_PER_REQUEST)
            requests_made += 1
            days_fetched += 1
            
            for article in articles:
                text = f"{article['title']} {article['description']}"
                sentiment = calculate_sentiment(text, analyzer)
                
                all_news.append({
                    "date": date_str,
                    "uuid": article["uuid"],
                    "title": article["title"],
                    "description": article["description"],
                    "source": article["source"],
                    "categories": article["categories"],
                    "sentiment_compound": sentiment["compound"],
                    "sentiment_neg": sentiment["neg"],
                    "sentiment_neu": sentiment["neu"],
                    "sentiment_pos": sentiment["pos"]
                })
            
            if days_fetched % 50 == 0:
                print(f"  {date_str}... ({days_fetched} days, {len(all_news)} articles)")
            
            current_date -= timedelta(days=1)
            time.sleep(0.2)
        
        print(f"  Fetched {len(all_news)} articles over {days_fetched} days")
        
        total_needed = days_back - len(existing_dates)
        remaining = total_needed - days_fetched
        if remaining > 0:
            print(f"  {remaining} days remaining. Run again tomorrow.")
        
        return all_news
    
    except (RateLimitError, UsageLimitError) as e:
        print(f"  ERROR: {e}")
        raise


def aggregate_daily_data(dsa_rows, news_rows):
    dsa_daily = defaultdict(int)
    for row in dsa_rows:
        dsa_daily[row["date"]] += row["count"]
    
    news_daily = defaultdict(lambda: {"count": 0, "sentiments": [], "neg_scores": [], "pos_scores": []})
    
    if news_rows:
        for row in news_rows:
            date = row["date"]
            news_daily[date]["count"] += 1
            news_daily[date]["sentiments"].append(row["sentiment_compound"])
            news_daily[date]["neg_scores"].append(row["sentiment_neg"])
            news_daily[date]["pos_scores"].append(row["sentiment_pos"])
    
    all_dates = sorted(set(list(dsa_daily.keys()) + list(news_daily.keys())))
    
    combined = []
    for date in all_dates:
        dsa_count = dsa_daily.get(date, 0)
        news_data = news_daily.get(date, {"count": 0, "sentiments": [], "neg_scores": [], "pos_scores": []})
        
        sentiments = news_data["sentiments"] if news_data["sentiments"] else [0]
        neg_scores = news_data["neg_scores"] if news_data["neg_scores"] else [0]
        pos_scores = news_data["pos_scores"] if news_data["pos_scores"] else [0]
        
        combined.append({
            "date": date,
            "dsa_count": dsa_count,
            "news_count": news_data["count"],
            "sentiment_mean": np.mean(sentiments),
            "sentiment_min": np.min(sentiments),
            "sentiment_max": np.max(sentiments),
            "sentiment_std": np.std(sentiments) if len(sentiments) > 1 else 0,
            "sentiment_neg_mean": np.mean(neg_scores),
            "sentiment_pos_mean": np.mean(pos_scores),
            "negative_news_pct": sum(1 for s in sentiments if s < -0.05) / len(sentiments) if sentiments else 0,
            "positive_news_pct": sum(1 for s in sentiments if s > 0.05) / len(sentiments) if sentiments else 0
        })
    
    return combined


def plot_backfill_data(df, dsa_df, filename=BACKFILL_PLOT):
    print(f"\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    ax1 = axes[0]
    daily_totals = df.groupby('date')['dsa_count'].sum() if 'dsa_count' in df.columns else df.set_index('date')['dsa_count']
    ax1.fill_between(df['date'], df['dsa_count'], alpha=0.3, color='#e74c3c')
    ax1.plot(df['date'], df['dsa_count'], color='#e74c3c', linewidth=1)
    rolling_avg = df['dsa_count'].rolling(window=7, min_periods=1).mean()
    ax1.plot(df['date'], rolling_avg, color='#c0392b', linewidth=2, linestyle='--', label='7-day avg')
    ax1.set_ylabel('Total DSA Violations', fontsize=10)
    ax1.set_title('Total DSA Violations Over Time (3 Categories)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[1]
    colors = {'Risk for public security': '#e74c3c', 
              'Negative effects on civic discourse or elections': '#3498db',
              'Illegal or harmful speech': '#9b59b6'}
    
    if dsa_df is not None and len(dsa_df) > 0:
        pivot_df = dsa_df.pivot_table(index='date', columns='violation', values='count', aggfunc='sum').fillna(0)
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df.sort_index()
        
        for violation in TARGET_VIOLATIONS:
            if violation in pivot_df.columns:
                ax2.plot(pivot_df.index, pivot_df[violation], linewidth=1.5, 
                        label=violation[:30], color=colors.get(violation, '#333'))
        
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('DSA Violations by Category', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    ax3 = axes[2]
    if 'sentiment_mean' in df.columns:
        colors_sent = ['#27ae60' if s > 0 else '#e74c3c' for s in df['sentiment_mean']]
        ax3.bar(df['date'], df['sentiment_mean'], color=colors_sent, alpha=0.6, width=1)
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Avg Sentiment', fontsize=10)
        ax3.set_title('Daily News Sentiment', fontsize=12, fontweight='bold')
        ax3.set_ylim(-1, 1)
    else:
        ax3.text(0.5, 0.5, 'No sentiment data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax3.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("DSA DATA BACKFILL")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    errors, warnings = validate_config()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print_config()
    ensure_directories()
    
    try:
        news_rows = fetch_news_history(days_back=365)
    except (RateLimitError, UsageLimitError):
        print("\nNews fetching stopped due to API limits")
        news_rows = []
    
    if news_rows:
        df_news = pd.DataFrame(news_rows)
        df_news.to_csv(NEWS_CSV, index=False)
        print(f"  Saved {len(news_rows)} rows to {NEWS_CSV}")
    else:
        print("\nNo news data fetched")
        return
    
    print("\nFetching DSA data (365 days)...")
    dsa_rows = fetch_dsa_data(days_back=365)
    
    if dsa_rows:
        df_dsa = pd.DataFrame(dsa_rows)
        df_dsa.to_csv(DSA_CSV, index=False)
        print(f"  Saved {len(dsa_rows)} rows to {DSA_CSV}")
    else:
        print("  No DSA data fetched")
        return
    
    combined_data = aggregate_daily_data(dsa_rows, news_rows)
    df_combined = pd.DataFrame(combined_data)
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    df_combined.to_csv(COMBINED_CSV, index=False)
    print(f"  Saved combined data to {COMBINED_CSV}")
    
    plot_backfill_data(df_combined, df_dsa)
    
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  {DSA_CSV}")
    print(f"  {NEWS_CSV}")
    print(f"  {COMBINED_CSV}")
    print(f"  {BACKFILL_PLOT}")
    
    print(f"\nData summary:")
    print(f"  Date range: {df_combined['date'].min().strftime('%Y-%m-%d')} to {df_combined['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total days: {len(df_combined)}")
    print(f"  Total DSA reports: {df_combined['dsa_count'].sum():,.0f}")
    print(f"  Avg daily DSA: {df_combined['dsa_count'].mean():,.0f}")


if __name__ == "__main__":
    main()
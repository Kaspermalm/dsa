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
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    install("python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from config import (
    NEWS_API_TOKEN, NEWS_ARTICLES_PER_REQUEST, NEWS_DAILY_REQUESTS,
    NEWS_API_TOP, NEWS_API_ALL,
    CSV_DIR,
    DSA_CSV, NEWS_CSV, COMBINED_CSV,
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


def fetch_recent_dsa_data(days_back=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"  Fetching DSA data from {start_date.strftime('%Y-%m-%d')}...")
    
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
                            "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 1000}}, "Secondary": {"Top": {"Count": 60}}},
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
            return rows
        
        result = results[0].get("result", {})
        dsr = result.get("data", {}).get("dsr", {})
        data_shapes = dsr.get("DS", [])
        
        if not data_shapes:
            return rows
        
        ds = data_shapes[0]
        value_dicts = ds.get("ValueDicts", {})
        violations = value_dicts.get("D0", [])
        
        ph = ds.get("PH", [])
        if not ph:
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
        return []


def fetch_news_for_date(date_str, limit=None):
    if limit is None:
        limit = NEWS_ARTICLES_PER_REQUEST
    
    for endpoint in [NEWS_API_TOP, NEWS_API_ALL]:
        params = {
            "api_token": NEWS_API_TOKEN,
            "locale": "us",
            "language": "en",
            "published_on": date_str,
            "limit": limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 429:
                raise RateLimitError("Rate limit reached (HTTP 429)")
            
            if response.status_code != 200:
                continue
            
            data = response.json()
            
            if "error" in data:
                error_code = data["error"].get("code", "")
                error_msg = data["error"].get("message", str(data["error"]))
                
                if error_code == "rate_limit_reached":
                    raise RateLimitError(f"Rate limit reached: {error_msg}")
                elif error_code == "usage_limit_reached":
                    raise UsageLimitError(f"Daily usage limit reached: {error_msg}")
                elif error_code == "endpoint_access_restricted":
                    continue
                else:
                    continue
            
            articles = []
            if "data" in data and data["data"]:
                for article in data["data"]:
                    articles.append({
                        "uuid": article.get("uuid", ""),
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "source": article.get("source", ""),
                        "categories": ",".join(article.get("categories", [])),
                        "published_at": article.get("published_at", "")
                    })
            
            if articles:
                return articles
                
        except (RateLimitError, UsageLimitError):
            raise
        except:
            continue
    
    return []


def calculate_sentiment(text, analyzer):
    if not text or not isinstance(text, str):
        return {"compound": 0, "neg": 0, "neu": 0, "pos": 0}
    return analyzer.polarity_scores(text)


def fetch_recent_news(days_back=1):
    print(f"  Fetching news for today...")
    
    analyzer = SentimentIntensityAnalyzer()
    all_news = []
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        articles = fetch_news_for_date(date_str)
        
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
        
        print(f"  Fetched {len(all_news)} news articles for {date_str}")
        return all_news
    
    except (RateLimitError, UsageLimitError) as e:
        print(f"  ERROR: {e}")
        raise


def aggregate_daily_data(dsa_rows, news_rows):
    dsa_daily = defaultdict(int)
    for row in dsa_rows:
        dsa_daily[row["date"]] += row["count"]
    
    news_daily = defaultdict(lambda: {"count": 0, "sentiments": [], "neg_scores": [], "pos_scores": []})
    
    if news_rows and len(news_rows) > 0:
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


def update_csv_files(dsa_rows, news_rows, new_daily_data):
    if os.path.exists(DSA_CSV):
        existing_dsa = pd.read_csv(DSA_CSV)
        new_dsa = pd.DataFrame(dsa_rows)
        
        existing_keys = set(zip(existing_dsa['date'], existing_dsa['violation']))
        new_dsa_filtered = new_dsa[~new_dsa.apply(lambda x: (x['date'], x['violation']) in existing_keys, axis=1)]
        
        if len(new_dsa_filtered) > 0:
            combined_dsa = pd.concat([existing_dsa, new_dsa_filtered], ignore_index=True)
            combined_dsa.to_csv(DSA_CSV, index=False)
            print(f"  Added {len(new_dsa_filtered)} new DSA records")
        else:
            print(f"  No new DSA records to add")
    else:
        pd.DataFrame(dsa_rows).to_csv(DSA_CSV, index=False)
        print(f"  Created {DSA_CSV} with {len(dsa_rows)} records")
    
    if news_rows and len(news_rows) > 0:
        if os.path.exists(NEWS_CSV):
            existing_news = pd.read_csv(NEWS_CSV)
            new_news = pd.DataFrame(news_rows)
            
            existing_uuids = set(existing_news['uuid'])
            new_news_filtered = new_news[~new_news['uuid'].isin(existing_uuids)]
            
            if len(new_news_filtered) > 0:
                combined_news = pd.concat([existing_news, new_news_filtered], ignore_index=True)
                combined_news.to_csv(NEWS_CSV, index=False)
                print(f"  Added {len(new_news_filtered)} new news articles")
            else:
                print(f"  No new news articles to add")
        else:
            pd.DataFrame(news_rows).to_csv(NEWS_CSV, index=False)
            print(f"  Created {NEWS_CSV} with {len(news_rows)} records")
    else:
        print(f"  No news articles fetched")
    
    if new_daily_data and len(new_daily_data) > 0:
        if os.path.exists(COMBINED_CSV):
            existing_combined = pd.read_csv(COMBINED_CSV)
            existing_combined['date'] = pd.to_datetime(existing_combined['date']).dt.strftime('%Y-%m-%d')
            new_combined = pd.DataFrame(new_daily_data)
            
            existing_dates = set(existing_combined['date'])
            
            updated_count = 0
            new_rows = []
            for _, row in new_combined.iterrows():
                row_date = row['date']
                if row_date in existing_dates:
                    mask = existing_combined['date'] == row_date
                    for col in row.index:
                        if col in existing_combined.columns:
                            existing_combined.loc[mask, col] = row[col]
                    updated_count += 1
                else:
                    new_rows.append(row)
            
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                existing_combined = pd.concat([existing_combined, new_rows_df], ignore_index=True)
            
            existing_combined['date'] = pd.to_datetime(existing_combined['date'])
            existing_combined = existing_combined.sort_values('date').reset_index(drop=True)
            existing_combined.to_csv(COMBINED_CSV, index=False)
            print(f"  Updated {COMBINED_CSV} ({updated_count} updated, {len(new_rows)} new)")
        else:
            df = pd.DataFrame(new_daily_data)
            df.to_csv(COMBINED_CSV, index=False)
            print(f"  Created {COMBINED_CSV} with {len(new_daily_data)} records")
    else:
        print(f"  No daily data to update")


def main():
    print("=" * 60)
    print("DSA DAILY UPDATE")
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
    
    DSA_LOOKBACK_DAYS = 7
    
    print("\nFetching recent DSA data...")
    dsa_rows = fetch_recent_dsa_data(days_back=DSA_LOOKBACK_DAYS)
    
    print("\nFetching today's news...")
    try:
        news_rows = fetch_recent_news()
    except (RateLimitError, UsageLimitError):
        print("  News fetching failed due to API limits")
        news_rows = []
    
    print("\nAggregating data...")
    new_daily_data = aggregate_daily_data(dsa_rows, news_rows)
    
    print("\nUpdating CSV files...")
    update_csv_files(dsa_rows, news_rows, new_daily_data)
    
    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
        df['date'] = pd.to_datetime(df['date'])
        print(f"\nCurrent data status:")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Latest DSA count: {df.iloc[-1]['dsa_count']:,.0f}")


if __name__ == "__main__":
    main()
"""External market signal: weekly search-interest index (Google Trends via PyTrends).

Provides:
* ``fetch_live_trends``   - Fresh data from Google Trends with timeout, exponential backoff,
                            fashion keyword fallbacks, and weekly resampling.
* ``load_trends``         - Robust loader prioritizing live data when requested, with graceful
                            fallback to verified cache or calibrated simulation.
* ``simulated_trends``    - Deterministic benchmark used for synthetic data and offline operation.
* Helper status functions - Cleanly distinguish live, cached, and simulated data sources.

``Trend_Index`` of a garment = mean of the interest series of its pattern group and its fit.
"""
from __future__ import annotations

import json
import logging
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as C

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ exceptions
class GoogleTrendsError(Exception):
    """Base exception for Google Trends data fetching."""


class GoogleTrendsRateLimitError(GoogleTrendsError):
    """Raised when Google Trends rate limits requests (HTTP 429)."""


class GoogleTrendsNoDataError(GoogleTrendsError):
    """Raised when Google Trends returns no data for search terms."""


# ------------------------------------------------------------------ metadata helpers
def _metadata_path() -> Path:
    return getattr(C, "TREND_METADATA_PATH", C.EXTERNAL_DIR / "google_trends_metadata.json")


def get_trend_metadata() -> dict[str, Any]:
    """Return stored metadata regarding the latest trends fetch / cache state."""
    path = _metadata_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "status": "uninitialized",
        "source": "simulated",
        "last_updated": None,
        "geo": C.TREND_GEO,
        "timeframe": C.TREND_TIMEFRAME,
        "terms_used": {},
        "note": "Using default benchmark dataset."
    }


def save_trend_metadata(meta: dict[str, Any]) -> None:
    """Save fetch metadata to external directory."""
    path = _metadata_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save trend metadata: %s", exc)


def is_live(table: pd.DataFrame) -> bool:
    """Check if the provided trend table contains fresh live data."""
    if table is None or table.empty or "source" not in table.columns:
        return False
    return str(table["source"].iloc[0]).lower() in ("live", "google_trends")


def is_cached(table: pd.DataFrame) -> bool:
    """Check if the provided trend table is from saved cache."""
    if table is None or table.empty or "source" not in table.columns:
        return False
    return str(table["source"].iloc[0]).lower() in ("cached", "google_trends_cached")


def is_simulated(table: pd.DataFrame) -> bool:
    """Check if the provided trend table is simulated benchmark data."""
    if table is None or table.empty or "source" not in table.columns:
        return True
    return str(table["source"].iloc[0]).lower() == "simulated"


def get_source_label(table: pd.DataFrame) -> str:
    """Human-readable data source label for retail presentation."""
    if is_live(table):
        return "Live Google Trends"
    if is_cached(table):
        meta = get_trend_metadata()
        updated = meta.get("last_updated_human") or meta.get("last_updated", "")
        return f"Cached Google Trends ({updated})" if updated else "Cached Google Trends"
    return "Market Baseline (Simulated)"


# ------------------------------------------------------------------ simulated benchmark
def simulated_trends(n_weeks: int = 13, seed: int = C.SEED) -> pd.DataFrame:
    """Deterministic market baseline used when live trends are unavailable."""
    rng = np.random.default_rng(seed + 7)
    t = np.arange(n_weeks)
    out = {}
    for key in C.TREND_KEYWORDS:
        base = 50 + 14 * np.sin(2 * np.pi * t / max(n_weeks, 1) + rng.uniform(0, 2 * np.pi))
        walk = np.cumsum(rng.normal(0, 4, n_weeks))
        series = base + (walk - walk.mean()) + rng.normal(0, 3, n_weeks)
        out[key] = np.clip(series, 5, 100).round(1)
    df = pd.DataFrame(out)
    df.insert(0, "week", t)
    df.insert(1, "date", (pd.Timestamp(C.STUDY_START) + pd.to_timedelta(t * 7, unit="D")).strftime("%Y-%m-%d"))
    df["source"] = "simulated"
    return df


# ------------------------------------------------------------------ live trends
def _fetch_keyword_series(
    pt: Any,
    key: str,
    terms: list[str],
    geo: str,
    timeframe: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    sleep: float = 2.0,
) -> tuple[pd.Series, str]:
    """Fetch interest series for a single key, trying primary term and fallbacks with backoff."""
    last_error: Exception | None = None
    rate_limited = False

    for term in terms:
        for attempt in range(max_retries):
            try:
                # Polite pacing before hitting endpoint
                time.sleep(sleep + random.uniform(0.3, 0.8))

                pt.build_payload([term], timeframe=timeframe, geo=geo)
                data = pt.interest_over_time()

                if data is None or data.empty or term not in data:
                    # Keyword returned no volume: try fallback candidate
                    last_error = GoogleTrendsNoDataError(f"Google Trends returned no data for '{term}'")
                    break

                raw_series = data[term].astype(float)

                # Resample daily data to weekly if required
                if len(raw_series) > 30 and hasattr(data.index, "resample"):
                    weekly = data[[term]].resample("W-SUN").mean()[term].round(1)
                else:
                    weekly = raw_series.round(1)

                if weekly.empty or weekly.isna().all():
                    break

                # Successfully retrieved valid data
                return weekly, term

            except Exception as exc:
                err_msg = str(exc)
                if "429" in err_msg:
                    rate_limited = True
                    wait_time = backoff_base * (2 ** attempt) + random.uniform(1.0, 2.5)
                    logger.warning("Google Trends rate-limited (429) for '%s' (attempt %d/%d). Backing off %.1fs...",
                                   term, attempt + 1, max_retries, wait_time)
                    time.sleep(wait_time)
                    last_error = GoogleTrendsRateLimitError(f"Rate limited on '{term}': {exc}")
                else:
                    logger.warning("Error fetching '%s' (attempt %d/%d): %s", term, attempt + 1, max_retries, exc)
                    last_error = exc
                    time.sleep(backoff_base + random.uniform(0.5, 1.0))

    if rate_limited:
        raise GoogleTrendsRateLimitError(f"Google Trends rate limit active for key '{key}'. {last_error}")
    raise GoogleTrendsNoDataError(f"No valid data returned for '{key}' across candidates {terms}: {last_error}")


def fetch_live_trends(
    geo: str = C.TREND_GEO,
    timeframe: str = C.TREND_TIMEFRAME,
    timeout: tuple[int, int] = (10, 25),
    max_retries: int = 2,
    sleep: float = 2.0,
) -> pd.DataFrame:
    """Download weekly interest for fashion keywords with PyTrends with retry and fallbacks.

    Raises:
        GoogleTrendsRateLimitError: If Google returns 429 and retries are exhausted.
        GoogleTrendsError: If fetching fails due to network, package, or data issues.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise GoogleTrendsError("The pytrends package is not installed. Run `pip install pytrends`.") from exc

    # Custom browser User-Agent headers to reduce false-positive bot-blocking
    requests_args = {
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        }
    }

    try:
        pt = TrendReq(hl="en-IN", tz=330, timeout=timeout, requests_args=requests_args)
    except Exception as exc:
        raise GoogleTrendsError(f"Failed to initialize PyTrends client: {exc}") from exc

    fallbacks_map = getattr(C, "TREND_KEYWORD_FALLBACKS", {})
    frames: dict[str, pd.Series] = {}
    terms_used: dict[str, str] = {}
    dates_index: pd.Index | None = None
    rate_limited_count = 0

    for key, primary_term in C.TREND_KEYWORDS.items():
        candidates = [primary_term] + fallbacks_map.get(key, [])
        # Deduplicate while preserving order
        candidates = list(dict.fromkeys(candidates))

        try:
            series, term_chosen = _fetch_keyword_series(
                pt, key, candidates, geo=geo, timeframe=timeframe,
                max_retries=max_retries, sleep=sleep
            )
            frames[key] = series.reset_index(drop=True)
            terms_used[key] = term_chosen
            if dates_index is None:
                dates_index = series.index
        except GoogleTrendsRateLimitError:
            rate_limited_count += 1
            if rate_limited_count >= 2:
                # Persistent 429: abort further hammering to protect client IP
                raise GoogleTrendsRateLimitError(
                    "Google Trends is currently rate-limiting queries (HTTP 429). "
                    "Rate limit is temporary; cached or benchmark data will be used."
                )
        except Exception as exc:
            logger.warning("Could not fetch trends for %s: %s", key, exc)

    # Check if we got sufficient keywords
    if not frames:
        if rate_limited_count > 0:
            raise GoogleTrendsRateLimitError("Google Trends returned HTTP 429 for all requested keywords.")
        raise GoogleTrendsNoDataError("Unable to retrieve search volume for any fashion keywords.")

    # If some keywords succeeded but 1 or 2 failed, fill missing gracefully
    if len(frames) < len(C.TREND_KEYWORDS):
        ref_len = len(next(iter(frames.values())))
        for key in C.TREND_KEYWORDS:
            if key not in frames:
                # Estimate missing series as average of available series
                available = pd.DataFrame(frames)
                frames[key] = available.mean(axis=1).round(1)
                terms_used[key] = f"interpolated ({C.TREND_KEYWORDS[key]})"

    # Assemble final weekly dataframe
    df = pd.DataFrame(frames)
    df.insert(0, "week", range(len(df)))

    if dates_index is not None and hasattr(dates_index, "strftime"):
        df.insert(1, "date", [d.strftime("%Y-%m-%d") for d in dates_index[:len(df)]])
    else:
        df.insert(1, "date", (pd.Timestamp(C.STUDY_START) + pd.to_timedelta(np.arange(len(df)) * 7, unit="D")).strftime("%Y-%m-%d"))

    df["source"] = "live"

    # Save metadata
    now_utc = datetime.now(timezone.utc).isoformat()
    now_human = datetime.now().strftime("%d %b %Y, %H:%M")
    save_trend_metadata({
        "status": "connected",
        "source": "live",
        "last_updated": now_utc,
        "last_updated_human": now_human,
        "geo": geo,
        "timeframe": timeframe,
        "terms_used": terms_used,
        "note": "Fresh live data fetched via PyTrends."
    })

    return df


# ------------------------------------------------------------------ robust loader
def load_trends(
    live: bool = False,
    n_weeks: int = 13,
    save: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return the weekly trend table.

    Order of operations:
    1. If `live=True` or `force_refresh=True`: attempt fresh fetch from PyTrends.
       On success: caches result to `C.TRENDS_PATH` and returns with source="live".
       On failure (rate limit, network): falls back to cached data or simulated baseline.
    2. If `live=False`: loads from cache file if present (source="cached").
    3. If cache does not exist: returns deterministic calibrated simulation (source="simulated").
    """
    if live or force_refresh:
        try:
            df = fetch_live_trends()
            if save:
                try:
                    C.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
                    # When writing to disk, write source="cached" so subsequent reads know it is cached
                    save_df = df.copy()
                    save_df["source"] = "cached"
                    save_df.to_csv(C.TRENDS_PATH, index=False, encoding="utf-8")
                except Exception as save_err:
                    logger.warning("Could not save trend cache to %s: %s", C.TRENDS_PATH, save_err)
            return df
        except GoogleTrendsRateLimitError as rle:
            warnings.warn(f"Google Trends rate limited: {rle}")
            logger.info("Using cached/baseline trends due to Google Trends rate limiting.")
        except Exception as exc:
            warnings.warn(f"Live Google Trends query failed ({exc}); using cached/baseline values.")
            logger.info("Falling back from live trends to cache/baseline: %s", exc)

    # Check for cached dataset on disk
    if C.TRENDS_PATH.exists():
        try:
            cached_df = pd.read_csv(C.TRENDS_PATH)
            # Verify required columns are present
            if all(k in cached_df.columns for k in C.TREND_KEYWORDS):
                cached_df["source"] = "cached"
                return cached_df
        except Exception as read_err:
            logger.warning("Could not read cached trends from %s: %s", C.TRENDS_PATH, read_err)

    # Calibrated fallback baseline
    return simulated_trends(n_weeks)


# ------------------------------------------------------------------ helpers for product scoring
def trend_index(pattern_group: str, fit: str, table: pd.DataFrame, week: int | None = None) -> float:
    """Mean of pattern-group and fit interest at ``week`` (default: latest week)."""
    row = table.iloc[-1] if week is None else table.iloc[min(int(week), len(table) - 1)]
    return float(np.mean([row[f"pattern:{pattern_group}"], row[f"fit:{fit}"]]))


def default_trend_index(pattern: str, fit: str) -> tuple[float, str]:
    """Latest trend value for a garment's pattern/fit, plus the data source label."""
    table = load_trends(live=False)
    grp = C.PATTERN_TO_GROUP[pattern]
    return trend_index(grp, fit, table), str(table["source"].iloc[0])

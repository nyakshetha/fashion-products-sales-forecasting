"""Unit tests for src/trends.py: Google Trends fetching, caching, and fallback handling."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config as C
from src import trends as T


def test_simulated_trends_structure():
    df = T.simulated_trends(13)
    assert len(df) == 13
    assert "week" in df.columns
    assert "date" in df.columns
    assert "source" in df.columns
    assert (df["source"] == "simulated").all()
    for key in C.TREND_KEYWORDS:
        assert key in df.columns
        assert df[key].between(0, 100).all()


def test_simulated_trends_deterministic():
    t1 = T.simulated_trends(13, seed=C.SEED)
    t2 = T.simulated_trends(13, seed=C.SEED)
    pd.testing.assert_frame_equal(t1, t2)


def test_trend_index_calculation():
    df = T.simulated_trends(13)
    idx_latest = T.trend_index("solid", "Slim", df)
    expected = float(np.mean([df["pattern:solid"].iloc[-1], df["fit:Slim"].iloc[-1]]))
    assert abs(idx_latest - expected) < 1e-4

    idx_week2 = T.trend_index("texture", "Ultra Slim", df, week=2)
    expected_w2 = float(np.mean([df["pattern:texture"].iloc[2], df["fit:Ultra Slim"].iloc[2]]))
    assert abs(idx_week2 - expected_w2) < 1e-4


def test_default_trend_index():
    val, src = T.default_trend_index("Solid", "Slim")
    assert 0 <= val <= 100
    assert src in ("simulated", "cached", "live", "google_trends")


def test_source_identification_helpers():
    sim = pd.DataFrame({"source": ["simulated"]})
    assert T.is_simulated(sim)
    assert not T.is_live(sim)
    assert not T.is_cached(sim)
    assert "Simulated" in T.get_source_label(sim)

    live = pd.DataFrame({"source": ["live"]})
    assert T.is_live(live)
    assert not T.is_simulated(live)
    assert not T.is_cached(live)
    assert "Live" in T.get_source_label(live)

    cached = pd.DataFrame({"source": ["cached"]})
    assert T.is_cached(cached)
    assert not T.is_live(cached)
    assert not T.is_simulated(cached)
    assert "Cached" in T.get_source_label(cached)


def test_metadata_read_write(tmp_path, monkeypatch):
    meta_file = tmp_path / "test_meta.json"
    monkeypatch.setattr(T, "_metadata_path", lambda: meta_file)

    meta = {
        "status": "connected",
        "source": "live",
        "last_updated": "2026-11-20T23:00:00",
        "last_updated_human": "20 Nov 2026, 23:00",
        "terms_used": {"pattern:solid": "plain shirt"}
    }
    T.save_trend_metadata(meta)
    loaded = T.get_trend_metadata()
    assert loaded["source"] == "live"
    assert loaded["terms_used"]["pattern:solid"] == "plain shirt"


def test_load_trends_cache_interaction(tmp_path, monkeypatch):
    cache_file = tmp_path / "google_trends_weekly.csv"
    monkeypatch.setattr(C, "TRENDS_PATH", cache_file)

    # 1. When cache does not exist, returns simulated
    df = T.load_trends(live=False, n_weeks=10)
    assert len(df) == 10
    assert df["source"].iloc[0] == "simulated"

    # 2. Save a dummy cache
    sim = T.simulated_trends(12)
    sim.to_csv(cache_file, index=False)

    # 3. When cache exists, returns cached data
    loaded = T.load_trends(live=False)
    assert len(loaded) == 12
    assert loaded["source"].iloc[0] == "cached"


def test_load_trends_rate_limit_graceful_fallback(tmp_path, monkeypatch):
    cache_file = tmp_path / "google_trends_weekly.csv"
    monkeypatch.setattr(C, "TRENDS_PATH", cache_file)

    # Mock fetch_live_trends to raise rate limit error
    with patch("src.trends.fetch_live_trends", side_effect=T.GoogleTrendsRateLimitError("429 Too Many Requests")):
        # Without cache -> falls back to simulated without raising
        df = T.load_trends(live=True, n_weeks=13)
        assert df["source"].iloc[0] == "simulated"
        assert len(df) == 13

        # With cache -> falls back to cached without raising
        sim = T.simulated_trends(15)
        sim.to_csv(cache_file, index=False)
        df_cached = T.load_trends(live=True)
        assert df_cached["source"].iloc[0] == "cached"
        assert len(df_cached) == 15


def test_keyword_coverage():
    """Verify that fashion keywords and fallbacks cover all expected garment categories."""
    expected_keys = {"pattern:solid", "pattern:texture", "pattern:geometric",
                     "pattern:abstract", "fit:Slim", "fit:Ultra Slim"}
    assert set(C.TREND_KEYWORDS.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(C.TREND_KEYWORDS[k], str)
        assert len(C.TREND_KEYWORDS[k]) > 0
        fallbacks = C.TREND_KEYWORD_FALLBACKS.get(k, [])
        assert isinstance(fallbacks, list)

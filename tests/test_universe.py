"""Tests for the universe building blocks. The bug targeted here is *silent*: a
failed ticker conversion drops a name from the universe without ever raising."""
import pandas as pd
import pytest

from src.universe import normalize_ticker, parse_constituents, snapshot_dates


@pytest.mark.parametrize("raw,expected", [
    ("BRK.B", "BRK-B"),        # Yahoo Finance convention
    ("BF.B", "BF-B"),
    (" nee ", "NEE"),          # whitespace + case
    ("AES[1]", "AES"),         # wiki footnote marker
    ("SO", "SO"),
])
def test_normalize_ticker(raw, expected):
    assert normalize_ticker(raw) == expected


def test_snapshot_dates_are_semiannual_and_bounded():
    d = snapshot_dates("2014-01-01", "2026-06-30", months=6)
    assert d[0] == pd.Timestamp("2014-01-01")
    assert d[-1] <= pd.Timestamp("2026-06-30")
    assert (d.to_series().diff().dropna().dt.days.between(180, 185)).all()


def test_parse_constituents_picks_table_and_renames_columns():
    # Minimal HTML mimicking an old revision ("Ticker symbol", not "Symbol")
    rows = "".join(
        f"<tr><td>T{i}</td><td>Co {i}</td><td>Utilities</td><td>Electric</td></tr>"
        for i in range(320)
    )
    html = ("<table><tr><th>Ticker symbol</th><th>Security</th>"
            "<th>GICS Sector</th><th>GICS Sub Industry</th></tr>" + rows + "</table>")
    out = parse_constituents(html)
    assert list(out.columns) == ["ticker", "name", "sector", "sub_industry"]
    assert len(out) == 320
    assert (out["sector"] == "Utilities").all()

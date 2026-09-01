from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


SIBLIS_URL = "https://siblisresearch.com/data/pe-ratios-by-country/"
MACROMICRO_URL = "https://www.macromicro.me/cross-country-database/pe-ratio"

SIBLIS_MARKETS = {
    "台灣｜加權指數": "Taiwan",
    "美國｜S&P 500": "United States",
    "日本｜日經225": "Japan",
    "韓國｜KOSPI": "South Korea",
    "中國｜上證指數": "China",
    "香港｜恆生指數": "Hong Kong",
    "德國｜DAX": "Germany",
    "英國｜FTSE 100": "United Kingdom",
    "法國｜CAC 40": "France",
    "澳洲｜ASX 200": "Australia",
}


def fetch_siblis_country_valuations(timeout: int = 25) -> tuple[pd.DataFrame, str]:
    """Read Siblis' public country table. Full daily/API history requires their subscription."""
    headers = {
        "User-Agent": "inventory-cycle-research-dashboard/1.0",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://siblisresearch.com/",
    }
    response = requests.get(SIBLIS_URL, headers=headers, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError("Siblis 公開頁面未找到估值表格")
    raw = tables[0].copy()
    required = {"Market", "Ratio"}
    if not required.issubset(raw.columns):
        raise ValueError("Siblis 公開表格格式已變更")
    date_columns = [column for column in raw.columns if column not in {"Market", "Calculated Using", "Ratio"}]
    if not date_columns:
        raise ValueError("Siblis 公開表格沒有日期欄")
    latest_column = max(date_columns, key=lambda value: pd.to_datetime(value, errors="coerce"))
    raw["Market"] = raw["Market"].ffill()
    raw[latest_column] = pd.to_numeric(raw[latest_column], errors="coerce")

    rows = []
    for app_market, source_market in SIBLIS_MARKETS.items():
        block = raw.loc[raw["Market"].eq(source_market)]
        values = dict(zip(block["Ratio"].astype(str).str.strip(), block[latest_column]))
        rows.append(
            {
                "市場": app_market,
                "資料日期": pd.to_datetime(latest_column).date().isoformat(),
                "TTM本益比": values.get("P/E (TTM)"),
                "Forward本益比": values.get("Forward P/E"),
                "TTM_EPS指數": values.get("EPS (TTM)*"),
                "來源": "Siblis Research 公開跨國估值表",
                "來源網址": SIBLIS_URL,
            }
        )
    return pd.DataFrame(rows), latest_column


def valuation_for_market(frame: pd.DataFrame, market: str) -> dict | None:
    match = frame.loc[frame["市場"].eq(market)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    if pd.isna(row.get("Forward本益比")):
        return None
    return row

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


PHASES = {
    "主動去庫存": {
        "quadrant": "需求改善、庫存下降",
        "signal": "復甦初期",
        "score": 74,
        "pe_multiplier": 1.05,
        "color": "#29B6A6",
        "explanation": "企業仍在清庫存，但需求動能已轉強；通常是獲利落底前後、估值先行修復的階段。",
    },
    "主動補庫存": {
        "quadrant": "需求改善、庫存上升",
        "signal": "擴張期",
        "score": 84,
        "pe_multiplier": 1.10,
        "color": "#3B82F6",
        "explanation": "訂單與需求擴張，企業主動增加生產與庫存；獲利能見度通常最佳，但後段需留意庫存增速過快。",
    },
    "被動補庫存": {
        "quadrant": "需求轉弱、庫存上升",
        "signal": "景氣後期",
        "score": 38,
        "pe_multiplier": 0.88,
        "color": "#F59E0B",
        "explanation": "需求已降溫，但企業尚未及時減產，庫存被動堆高；容易出現獲利下修與本益比壓縮。",
    },
    "被動去庫存": {
        "quadrant": "需求轉弱、庫存下降",
        "signal": "收縮期",
        "score": 25,
        "pe_multiplier": 0.80,
        "color": "#EF4444",
        "explanation": "需求與生產同步收縮，企業降低庫存以保現金；基本面壓力仍在，但後段可等待需求率先止跌。",
    },
}


@dataclass(frozen=True)
class CycleResult:
    phase: str
    score: int
    signal: str
    explanation: str
    demand_momentum: float
    inventory_momentum: float


def classify_cycle(
    demand_growth: float,
    inventory_growth: float,
    previous_demand_growth: float,
    previous_inventory_growth: float,
) -> CycleResult:
    """Classify by the direction of demand and inventory growth, not their absolute level."""
    demand_momentum = demand_growth - previous_demand_growth
    inventory_momentum = inventory_growth - previous_inventory_growth
    if demand_momentum >= 0 and inventory_momentum < 0:
        phase = "主動去庫存"
    elif demand_momentum >= 0 and inventory_momentum >= 0:
        phase = "主動補庫存"
    elif demand_momentum < 0 and inventory_momentum >= 0:
        phase = "被動補庫存"
    else:
        phase = "被動去庫存"

    meta = PHASES[phase]
    momentum_adjustment = np.clip(demand_momentum * 1.5 - inventory_momentum * 0.35, -12, 12)
    score = int(np.clip(round(meta["score"] + momentum_adjustment), 0, 100))
    return CycleResult(
        phase=phase,
        score=score,
        signal=meta["signal"],
        explanation=meta["explanation"],
        demand_momentum=float(demand_momentum),
        inventory_momentum=float(inventory_momentum),
    )


def fair_value_scenarios(
    current_price: float,
    forward_eps: float,
    historical_pe: float,
    discount_rate: float,
    anchor_rate: float,
    phase: str,
    rate_sensitivity: float = 6.0,
) -> pd.DataFrame:
    """Estimate scenario fair values using EPS × cycle/rate-adjusted P/E."""
    phase_multiplier = float(PHASES[phase]["pe_multiplier"])
    rate_gap = (discount_rate - anchor_rate) / 100
    rate_multiplier = math.exp(-rate_sensitivity * rate_gap)
    base_pe = historical_pe * phase_multiplier * rate_multiplier

    rows = []
    assumptions = {
        "保守": (0.88, 0.88),
        "中性": (1.00, 1.00),
        "樂觀": (1.10, 1.10),
    }
    for scenario, (eps_mult, pe_mult) in assumptions.items():
        eps = forward_eps * eps_mult
        pe = base_pe * pe_mult
        fair_value = eps * pe
        gap = (fair_value / current_price - 1) * 100 if current_price > 0 else np.nan
        rows.append(
            {
                "情境": scenario,
                "情境EPS": eps,
                "合理本益比": pe,
                "合理價格": fair_value,
                "相對現價%": gap,
            }
        )
    return pd.DataFrame(rows)


def enterprise_value_scenarios(
    current_price: float,
    forward_ebitda_per_share: float,
    historical_ev_ebitda: float,
    net_debt_per_share: float,
    discount_rate: float,
    anchor_rate: float,
    phase: str,
    rate_sensitivity: float = 5.0,
) -> pd.DataFrame:
    """Estimate equity value per share from EV/EBITDA, then subtract net debt per share."""
    phase_multiplier = float(PHASES[phase]["pe_multiplier"])
    rate_gap = (discount_rate - anchor_rate) / 100
    rate_multiplier = math.exp(-rate_sensitivity * rate_gap)
    base_multiple = historical_ev_ebitda * phase_multiplier * rate_multiplier
    assumptions = {
        "保守": (0.88, 0.90),
        "中性": (1.00, 1.00),
        "樂觀": (1.10, 1.10),
    }
    rows = []
    for scenario, (ebitda_mult, multiple_mult) in assumptions.items():
        ebitda_per_share = forward_ebitda_per_share * ebitda_mult
        ev_multiple = base_multiple * multiple_mult
        enterprise_value_per_share = ebitda_per_share * ev_multiple
        equity_value_per_share = max(enterprise_value_per_share - net_debt_per_share, 0)
        gap = (equity_value_per_share / current_price - 1) * 100 if current_price > 0 else np.nan
        rows.append(
            {
                "情境": scenario,
                "情境EBITDA/股": ebitda_per_share,
                "合理EV/EBITDA": ev_multiple,
                "每股企業價值": enterprise_value_per_share,
                "淨負債/股": net_debt_per_share,
                "合理價格": equity_value_per_share,
                "相對現價%": gap,
            }
        )
    return pd.DataFrame(rows)


def blend_valuations(pe_values: pd.DataFrame, ev_values: pd.DataFrame, pe_weight: float) -> pd.DataFrame:
    weight = float(np.clip(pe_weight, 0, 1))
    merged = pe_values[["情境", "合理價格"]].rename(columns={"合理價格": "P/E合理價"}).merge(
        ev_values[["情境", "合理價格"]].rename(columns={"合理價格": "EV合理價"}), on="情境"
    )
    merged["P/E權重"] = weight
    merged["EV權重"] = 1 - weight
    merged["綜合合理價"] = merged["P/E合理價"] * weight + merged["EV合理價"] * (1 - weight)
    return merged


def overall_market_score(cycle_score: int, valuation_gap: float, discount_rate: float, anchor_rate: float) -> int:
    valuation_component = float(np.clip(valuation_gap, -40, 40)) * 0.35
    rate_penalty = max(discount_rate - anchor_rate, 0) * 3.0
    return int(np.clip(round(cycle_score + valuation_component - rate_penalty), 0, 100))


def score_label(score: int) -> tuple[str, str]:
    if score >= 75:
        return "偏多", "需求、循環與估值的組合相對有利"
    if score >= 55:
        return "中性偏多", "仍有上行條件，但報酬空間需看估值"
    if score >= 40:
        return "中性觀望", "多空訊號混合，宜等待需求或估值改善"
    return "偏保守", "循環或估值壓力較高，先重視風險控管"


def normalize_uploaded(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "日期": "Date",
        "date": "Date",
        "需求年增率": "DemandGrowth",
        "demand_growth": "DemandGrowth",
        "營收年增率": "DemandGrowth",
        "庫存年增率": "InventoryGrowth",
        "inventory_growth": "InventoryGrowth",
        "存貨年增率": "InventoryGrowth",
        "價格": "Price",
        "price": "Price",
    }
    out = frame.rename(columns={column: aliases.get(str(column).strip(), column) for column in frame.columns}).copy()
    required = {"Date", "DemandGrowth", "InventoryGrowth"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError("缺少必要欄位：" + "、".join(sorted(missing)))
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for column in ("DemandGrowth", "InventoryGrowth", "Price"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["Date", "DemandGrowth", "InventoryGrowth"]).sort_values("Date")


def classify_history(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["DemandMomentum"] = out["DemandGrowth"].diff()
    out["InventoryMomentum"] = out["InventoryGrowth"].diff()
    conditions = [
        (out["DemandMomentum"] >= 0) & (out["InventoryMomentum"] < 0),
        (out["DemandMomentum"] >= 0) & (out["InventoryMomentum"] >= 0),
        (out["DemandMomentum"] < 0) & (out["InventoryMomentum"] >= 0),
        (out["DemandMomentum"] < 0) & (out["InventoryMomentum"] < 0),
    ]
    out["Phase"] = np.select(conditions, list(PHASES), default="資料不足")
    return out


# Streamlit Community Cloud may auto-select this first Python file as the entry
# point. In that case, hand off rendering to the actual dashboard without
# affecting normal imports from streamlit_app.py.
if __name__ == "__main__":
    import streamlit_app  # noqa: F401, E402

from __future__ import annotations

from datetime import date
from io import StringIO
import zlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from inventory_model import (
    PHASES,
    classify_cycle,
    classify_history,
    blend_valuations,
    enterprise_value_scenarios,
    fair_value_scenarios,
    normalize_uploaded,
    overall_market_score,
    score_label,
)
from valuation_sources import MACROMICRO_URL, SIBLIS_URL, fetch_siblis_country_valuations, valuation_for_market


st.set_page_config(page_title="庫存循環市場判讀", page_icon="🔄", layout="wide")

MARKETS = {
    "自訂股票／公司": {"ticker": "", "price": 100.0, "eps": 5.0, "pe": 18.0, "demand": 8.0, "inventory": 5.0, "prev_demand": 6.0, "prev_inventory": 6.0, "ebitda_ps": 8.0, "net_debt_ps": 10.0, "ev_multiple": 12.0},
    "台灣｜加權指數": {"ticker": "^TWII", "price": 46454.2, "eps": 2227.8, "pe": 17.0, "demand": 11.5, "inventory": 4.0, "prev_demand": 9.0, "prev_inventory": 5.2},
    "美國｜S&P 500": {"ticker": "^GSPC", "price": 6500.0, "eps": 305.0, "pe": 20.0, "demand": 4.0, "inventory": 2.5, "prev_demand": 3.2, "prev_inventory": 2.0},
    "日本｜日經225": {"ticker": "^N225", "price": 66682.9, "eps": 3020.4, "pe": 19.0, "demand": 3.0, "inventory": 1.5, "prev_demand": 2.0, "prev_inventory": 2.1},
    "韓國｜KOSPI": {"ticker": "^KS11", "price": 6839.2, "eps": 1028.3, "pe": 6.5, "demand": 12.0, "inventory": 8.0, "prev_demand": 14.0, "prev_inventory": 6.0},
    "中國｜上證指數": {"ticker": "000001.SS", "price": 3961.5, "eps": 282.8, "pe": 13.0, "demand": 4.5, "inventory": 3.0, "prev_demand": 5.1, "prev_inventory": 2.4},
    "香港｜恆生指數": {"ticker": "^HSI", "price": 25000.0, "eps": 1650.0, "pe": 10.5, "demand": 3.0, "inventory": 2.0, "prev_demand": 2.2, "prev_inventory": 2.8},
    "德國｜DAX": {"ticker": "^GDAXI", "price": 24500.0, "eps": 1420.0, "pe": 15.5, "demand": 1.8, "inventory": 2.5, "prev_demand": 1.2, "prev_inventory": 3.1},
    "英國｜FTSE 100": {"ticker": "^FTSE", "price": 9200.0, "eps": 610.0, "pe": 13.5, "demand": 1.5, "inventory": 1.8, "prev_demand": 1.0, "prev_inventory": 2.2},
    "法國｜CAC 40": {"ticker": "^FCHI", "price": 8100.0, "eps": 520.0, "pe": 14.5, "demand": 1.6, "inventory": 2.0, "prev_demand": 1.1, "prev_inventory": 2.5},
    "澳洲｜ASX 200": {"ticker": "^AXJO", "price": 8900.0, "eps": 485.0, "pe": 17.0, "demand": 2.4, "inventory": 2.1, "prev_demand": 2.0, "prev_inventory": 2.6},
    "巴西｜Bovespa": {"ticker": "^BVSP", "price": 140000.0, "eps": 11200.0, "pe": 10.5, "demand": 3.2, "inventory": 2.7, "prev_demand": 2.6, "prev_inventory": 3.3},
}


@st.cache_data(ttl=1800, show_spinner=False)
def market_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame()
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index().rename(columns={"Datetime": "Date"})
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.tz_localize(None)
        return data[["Date", "Close"]].dropna().sort_values("Date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl="12h", max_entries=2, show_spinner=False)
def public_valuation_data() -> tuple[pd.DataFrame, str]:
    try:
        frame, latest_column = fetch_siblis_country_valuations()
        return frame, f"Siblis 更新日 {latest_column}"
    except Exception as exc:
        return pd.DataFrame(), f"Siblis 暫時無法讀取：{type(exc).__name__}"


def demo_history(seed: int, months: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="MS")
    angle = np.linspace(0, 4 * np.pi, months)
    demand = 4.0 + 7.0 * np.sin(angle) + rng.normal(0, 0.8, months)
    inventory = 3.0 + 6.0 * np.sin(angle - np.pi / 2) + rng.normal(0, 0.7, months)
    price = 100 * np.exp(np.cumsum(0.006 + demand / 500 + rng.normal(0, 0.025, months)))
    return pd.DataFrame({"Date": dates, "DemandGrowth": demand, "InventoryGrowth": inventory, "Price": price})


def market_seed(market: str) -> int:
    return zlib.crc32(market.encode("utf-8"))


def sync_latest_assumptions(
    frame: pd.DataFrame,
    previous_demand: float,
    previous_inventory: float,
    demand_growth: float,
    inventory_growth: float,
    current_price: float,
) -> pd.DataFrame:
    """Put form assumptions into the last two observations so every chart and phase updates."""
    out = frame.copy().sort_values("Date").reset_index(drop=True)
    if len(out) < 2:
        end = pd.Timestamp.today().normalize().replace(day=1)
        out = pd.DataFrame({
            "Date": [end - pd.offsets.MonthBegin(1), end],
            "DemandGrowth": [previous_demand, demand_growth],
            "InventoryGrowth": [previous_inventory, inventory_growth],
            "Price": [current_price, current_price],
        })
    else:
        out.loc[out.index[-2], ["DemandGrowth", "InventoryGrowth"]] = [previous_demand, previous_inventory]
        out.loc[out.index[-1], ["DemandGrowth", "InventoryGrowth"]] = [demand_growth, inventory_growth]
        if "Price" not in out:
            out["Price"] = np.nan
        out.loc[out.index[-1], "Price"] = current_price
    return classify_history(out)


def phase_strip(history: pd.DataFrame) -> alt.Chart:
    phase_data = history.dropna(subset=["Phase"]).copy()
    return (
        alt.Chart(phase_data)
        .mark_rect(opacity=0.22)
        .encode(
            x=alt.X("Date:T", title=None),
            color=alt.Color(
                "Phase:N",
                scale=alt.Scale(domain=list(PHASES), range=[PHASES[p]["color"] for p in PHASES]),
                legend=alt.Legend(title="庫存循環"),
            ),
            tooltip=[alt.Tooltip("Date:T", title="日期", format="%Y-%m"), alt.Tooltip("Phase:N", title="階段")],
        )
        .properties(height=65)
    )


def business_cycle_history(history: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent demand-based business-cycle proxy from the active history."""
    out = history.copy().sort_values("Date").reset_index(drop=True)
    out["DemandMomentum"] = out["DemandGrowth"].diff()
    conditions = [
        (out["DemandGrowth"] >= 0) & (out["DemandMomentum"] >= 0),
        (out["DemandGrowth"] >= 0) & (out["DemandMomentum"] < 0),
        (out["DemandGrowth"] < 0) & (out["DemandMomentum"] < 0),
        (out["DemandGrowth"] < 0) & (out["DemandMomentum"] >= 0),
    ]
    out["BusinessPhase"] = np.select(conditions, ["擴張", "放緩", "收縮", "復甦"], default="資料不足")
    out["BusinessScore"] = np.clip(50 + out["DemandGrowth"] * 2 + out["DemandMomentum"] * 4, 0, 100)
    return out


def cycle_framework_chart(current_phase: str) -> alt.Chart:
    """Draw the four-stage demand/inventory framework as a conceptual matrix."""
    quadrants = pd.DataFrame(
        [
            {"x1": -1, "x2": 0, "y1": 0, "y2": 1, "x": -0.5, "y": 0.63, "phase": "主動去庫存", "position": "復甦初期", "reading": "需求改善、庫存下降；股價常領先基本面回升"},
            {"x1": 0, "x2": 1, "y1": 0, "y2": 1, "x": 0.5, "y": 0.63, "phase": "主動補庫存", "position": "擴張期", "reading": "需求改善、庫存增加；訂單與獲利通常較強"},
            {"x1": 0, "x2": 1, "y1": -1, "y2": 0, "x": 0.5, "y": -0.37, "phase": "被動補庫存", "position": "放緩／景氣後期", "reading": "需求轉弱、庫存增加；留意庫存與獲利下修"},
            {"x1": -1, "x2": 0, "y1": -1, "y2": 0, "x": -0.5, "y": -0.37, "phase": "被動去庫存", "position": "收縮期", "reading": "需求轉弱、庫存下降；後段等待下一輪復甦"},
        ]
    )
    quadrants["目前"] = quadrants["phase"].eq(current_phase)
    colors = alt.Scale(domain=list(PHASES), range=[PHASES[p]["color"] for p in PHASES])
    background = alt.Chart(quadrants).mark_rect(opacity=0.18, stroke="#64748B", strokeWidth=1).encode(
        x=alt.X("x1:Q", title="庫存動能　← 下降　　　　　　　　　增加 →", scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(values=[])),
        x2="x2:Q",
        y=alt.Y("y1:Q", title="需求動能　← 轉弱　　　　　　改善 →", scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(values=[])),
        y2="y2:Q",
        color=alt.Color("phase:N", scale=colors, legend=None),
        tooltip=[alt.Tooltip("phase:N", title="庫存循環"), alt.Tooltip("position:N", title="景氣位置"), alt.Tooltip("reading:N", title="市場解讀")],
    )
    phase_labels = alt.Chart(quadrants).mark_text(fontSize=20, fontWeight="bold", dy=-18).encode(
        x="x:Q", y="y:Q", text="phase:N", color=alt.Color("phase:N", scale=colors, legend=None)
    )
    position_labels = alt.Chart(quadrants).mark_text(fontSize=14, color="#94A3B8", dy=16).encode(
        x="x:Q", y="y:Q", text="position:N"
    )
    current = quadrants.loc[quadrants["目前"]].copy()
    marker = alt.Chart(current).mark_point(shape="diamond", filled=True, size=230, color="#FFFFFF", stroke="#111827", strokeWidth=2).encode(
        x="x:Q", y=alt.Y("y:Q")
    )
    marker_label = alt.Chart(current).mark_text(text="目前", fontSize=13, fontWeight="bold", color="#111827").encode(x="x:Q", y="y:Q")
    return (background + phase_labels + position_labels + marker + marker_label).properties(height=420)


st.title("庫存循環、股票獲利與企業價值")
st.caption("以庫存循環判斷景氣位置，再用 P/E 與 EV/EBITDA 雙模型估算股票合理價。所有結果均可調整假設，不是投資建議。")

with st.sidebar:
    st.header("分析設定")
    selected_market = st.selectbox("市場", list(MARKETS))
    defaults = MARKETS[selected_market]
    source_mode = st.segmented_control("資料方式", ["快速試算", "上傳歷史資料"], default="快速試算")
    live_price = st.toggle("嘗試取得最新市場價格", value=True)
    auto_valuation = st.toggle("讀取 Siblis 公開估值", value=True)
    st.divider()
    st.subheader("模型說明")
    st.markdown("需求與庫存動能交叉形成四階段。合理價同時計算 **EPS × P/E** 與 **EBITDA × EV/EBITDA－淨負債**。")

market_data = market_prices(defaults["ticker"]) if live_price else pd.DataFrame()
latest_market_price = float(market_data["Close"].iloc[-1]) if not market_data.empty else defaults["price"]
valuation_source_frame, valuation_source_note = public_valuation_data() if auto_valuation else (pd.DataFrame(), "已關閉自動估值")
source_valuation = valuation_for_market(valuation_source_frame, selected_market) if not valuation_source_frame.empty else None
if source_valuation:
    source_forward_pe = float(source_valuation["Forward本益比"])
    source_forward_eps = latest_market_price / source_forward_pe
else:
    source_forward_pe = float(defaults["pe"])
    source_forward_eps = float(defaults["eps"])

uploaded_history = None
if source_mode == "上傳歷史資料":
    with st.container(border=True):
        st.subheader("上傳月資料")
        st.write("必要欄位：日期、需求年增率、庫存年增率；價格欄位可省略。也接受 Date、demand_growth、inventory_growth、price。")
        upload = st.file_uploader("CSV 檔案", type=["csv"])
        sample = demo_history(market_seed(selected_market)).rename(columns={"Date": "日期", "DemandGrowth": "需求年增率", "InventoryGrowth": "庫存年增率", "Price": "價格"})
        st.download_button("下載範例 CSV", sample.to_csv(index=False).encode("utf-8-sig"), "庫存循環資料範例.csv", "text/csv")
        if upload is not None:
            try:
                uploaded_history = normalize_uploaded(pd.read_csv(upload))
                st.success(f"已讀取 {len(uploaded_history):,} 筆資料")
            except Exception as exc:
                st.error(str(exc))

history = classify_history(uploaded_history if uploaded_history is not None else demo_history(market_seed(selected_market)))
if uploaded_history is not None and len(history) >= 2:
    demand_default = float(history["DemandGrowth"].iloc[-1])
    inventory_default = float(history["InventoryGrowth"].iloc[-1])
    prev_demand_default = float(history["DemandGrowth"].iloc[-2])
    prev_inventory_default = float(history["InventoryGrowth"].iloc[-2])
else:
    demand_default, inventory_default = defaults["demand"], defaults["inventory"]
    prev_demand_default, prev_inventory_default = defaults["prev_demand"], defaults["prev_inventory"]

with st.expander("這些預設數字怎麼來的？"):
    st.markdown(
        "- **目前指數／股價**：開啟即時行情時，優先讀取 Yahoo Finance 最新收盤；失敗才使用程式內備援值。\n"
        "- **P/E**：開啟自動估值時，使用 Siblis Research 公開表格的 forward P/E；完整日資料與 API 屬訂閱服務。\n"
        "- **EPS**：Siblis 跨國表的 EPS 是基期 100 指數，不能直接乘上指數點位。本程式改用最新價格 ÷ Siblis forward P/E，反推同口徑的隱含 forward EPS。\n"
        "- **MacroMicro**：跨國頁會阻擋雲端自動請求，因此保留外部查核連結，不繞過登入或網站限制。\n"
        "- **需求、庫存、EBITDA 與淨負債**：仍是示範模型假設，請依實際資料覆寫，或上傳 CSV。\n"
        "- **下面圖表**：最後兩期會套用此表單的前期與本期數字；按重新計算後，循環階段、線圖與合理價會一起更新。"
    )

with st.container(border=True):
    s1, s2, s3, s4 = st.columns(4)
    if source_valuation:
        s1.metric("Siblis Forward P/E", f"{source_forward_pe:.2f}x", str(source_valuation["資料日期"]))
        s2.metric("Siblis TTM P/E", f"{float(source_valuation['TTM本益比']):.2f}x")
        s3.metric("Siblis TTM EPS 指數", f"{float(source_valuation['TTM_EPS指數']):.2f}", "基期100，非實際EPS")
        s4.metric("反推 Forward EPS", f"{source_forward_eps:,.2f}", "現價 ÷ Forward P/E")
        st.caption("來源：Siblis Research 公開跨國估值表。完整每日資料需使用其訂閱 API。")
    else:
        st.warning(f"{valuation_source_note}；此市場改用手動／備援估值假設。")
    link1, link2 = st.columns(2)
    link1.link_button("開啟 Siblis 原始資料", SIBLIS_URL, width="stretch")
    link2.link_button("開啟 MacroMicro 跨國本益比", MACROMICRO_URL, width="stretch")

market_key = f"{market_seed(selected_market)}_{'siblis' if source_valuation else 'manual'}"
with st.form(f"assumptions_{market_key}"):
    st.subheader("輸入本期假設")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        demand_growth = st.number_input("本期需求／營收年增率 %", value=float(demand_default), step=0.1, key=f"demand_{market_key}")
        previous_demand = st.number_input("前期需求／營收年增率 %", value=float(prev_demand_default), step=0.1, key=f"prev_demand_{market_key}")
    with c2:
        inventory_growth = st.number_input("本期庫存年增率 %", value=float(inventory_default), step=0.1, key=f"inventory_{market_key}")
        previous_inventory = st.number_input("前期庫存年增率 %", value=float(prev_inventory_default), step=0.1, key=f"prev_inventory_{market_key}")
    with c3:
        current_price = st.number_input("目前指數／股價", min_value=0.01, value=float(round(latest_market_price, 2)), step=1.0, key=f"price_{market_key}")
        forward_eps = st.number_input("未來 12 個月 EPS", min_value=0.01, value=float(source_forward_eps), step=1.0, key=f"eps_{market_key}")
    with c4:
        historical_pe = st.number_input("歷史／同業合理本益比", min_value=1.0, value=float(source_forward_pe), step=0.5, key=f"pe_{market_key}")
        discount_rate = st.number_input("目前折現率 %", min_value=0.1, value=7.5, step=0.25, key=f"discount_{market_key}")
        anchor_rate = st.number_input("長期基準折現率 %", min_value=0.1, value=7.0, step=0.25, key=f"anchor_{market_key}")
    st.markdown("**企業價值模型假設**")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        ebitda_per_share = st.number_input("未來12個月 EBITDA／股", min_value=0.01, value=float(defaults.get("ebitda_ps", defaults["eps"] * 1.6)), step=0.5, key=f"ebitda_{market_key}")
    with e2:
        historical_ev_multiple = st.number_input("合理 EV／EBITDA", min_value=1.0, value=float(defaults.get("ev_multiple", max(defaults["pe"] * 0.65, 4))), step=0.5, key=f"ev_multiple_{market_key}")
    with e3:
        net_debt_per_share = st.number_input("淨負債／股", value=float(defaults.get("net_debt_ps", 0.0)), step=0.5, help="總負債減現金，再除以流通股數；若淨現金請輸入負數。", key=f"net_debt_{market_key}")
    with e4:
        pe_weight_percent = st.slider("P/E 模型權重 %", min_value=0, max_value=100, value=60, step=5, key=f"pe_weight_{market_key}")
    submitted = st.form_submit_button("重新計算", type="primary", width="stretch")

cycle = classify_cycle(demand_growth, inventory_growth, previous_demand, previous_inventory)
history = sync_latest_assumptions(
    history, previous_demand, previous_inventory, demand_growth,
    inventory_growth, current_price
)
valuation = fair_value_scenarios(current_price, forward_eps, historical_pe, discount_rate, anchor_rate, cycle.phase)
enterprise_valuation = enterprise_value_scenarios(
    current_price, ebitda_per_share, historical_ev_multiple, net_debt_per_share,
    discount_rate, anchor_rate, cycle.phase
)
blended = blend_valuations(valuation, enterprise_valuation, pe_weight_percent / 100)
base_row = valuation.loc[valuation["情境"] == "中性"].iloc[0]
ev_base_row = enterprise_valuation.loc[enterprise_valuation["情境"] == "中性"].iloc[0]
blend_base_row = blended.loc[blended["情境"] == "中性"].iloc[0]
valuation_gap = (float(blend_base_row["綜合合理價"]) / current_price - 1) * 100
market_score = overall_market_score(cycle.score, valuation_gap, discount_rate, anchor_rate)
market_label, market_note = score_label(market_score)

st.subheader("目前判定")
top_left, top_right = st.columns(2)
top_left.metric("庫存循環", cycle.phase, cycle.signal, border=True)
top_right.metric("市場綜合分數", f"{market_score}/100", market_label, border=True)
bottom_left, bottom_right = st.columns(2)
bottom_left.metric(
    "雙模型綜合合理價",
    f"{blend_base_row['綜合合理價']:,.1f}",
    f"{valuation_gap:+.1f}% vs 現價",
    border=True,
)
bottom_right.metric(
    "兩種估值比較",
    f"P/E 法 {base_row['合理價格']:,.1f}",
    f"EV/EBITDA 法 {ev_base_row['合理價格']:,.1f}",
    border=True,
    delta_color="off",
)

left, right = st.columns([1.15, 0.85])
with left:
    with st.container(border=True):
        st.subheader("循環診斷")
        st.markdown(f"**{cycle.phase}｜{PHASES[cycle.phase]['quadrant']}**")
        st.write(cycle.explanation)
        st.write(f"需求動能為 **{cycle.demand_momentum:+.1f} 個百分點**，庫存動能為 **{cycle.inventory_momentum:+.1f} 個百分點**。")
        st.info(f"市場結論：{market_label}。{market_note}。")
with right:
    with st.container(border=True):
        st.subheader("雙模型合理價情境")
        display_blended = blended.copy()
        display_blended["相對現價%"] = (display_blended["綜合合理價"] / current_price - 1) * 100
        st.dataframe(
            display_blended,
            hide_index=True,
            width="stretch",
            column_config={
                "P/E合理價": st.column_config.NumberColumn(format="%,.1f"),
                "EV合理價": st.column_config.NumberColumn(format="%,.1f"),
                "綜合合理價": st.column_config.NumberColumn(format="%,.1f"),
                "P/E權重": st.column_config.NumberColumn(format="percent"),
                "EV權重": st.column_config.NumberColumn(format="percent"),
                "相對現價%": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )

tab1, tab2, tab3, tab4 = st.tabs(["循環與價格", "P/E 敏感度", "企業價值拆解", "方法與限制"])
with tab1:
    st.altair_chart(phase_strip(history), width="stretch")
    long = history.melt("Date", ["DemandGrowth", "InventoryGrowth"], var_name="指標", value_name="年增率")
    long["指標"] = long["指標"].map({"DemandGrowth": "需求年增率", "InventoryGrowth": "庫存年增率"})
    momentum_chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(x=alt.X("Date:T", title="日期"), y=alt.Y("年增率:Q", title="年增率（%）"), color=alt.Color("指標:N", title=None), tooltip=["Date:T", "指標:N", alt.Tooltip("年增率:Q", format=".2f")])
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(momentum_chart, width="stretch")
    price_frame = market_data.rename(columns={"Close": "市場價格"}) if not market_data.empty else history.rename(columns={"Price": "市場價格"})[["Date", "市場價格"]]
    current_point = pd.DataFrame({"Date": [pd.Timestamp.today().normalize()], "市場價格": [current_price]})
    price_frame = pd.concat([price_frame, current_point], ignore_index=True).dropna().sort_values("Date")
    fair_lines = pd.DataFrame({
        "情境": blended["情境"],
        "合理價": blended["綜合合理價"],
    })
    price_chart = alt.Chart(price_frame).mark_line(color="#70C7FF", strokeWidth=2).encode(
        x=alt.X("Date:T", title="日期"), y=alt.Y("市場價格:Q", scale=alt.Scale(zero=False)),
        tooltip=["Date:T", alt.Tooltip("市場價格:Q", format=",.2f")]
    )
    fair_chart = alt.Chart(fair_lines).mark_rule(strokeDash=[6, 4]).encode(
        y=alt.Y("合理價:Q"), color=alt.Color("情境:N", title="綜合合理價"),
        tooltip=["情境:N", alt.Tooltip("合理價:Q", format=",.2f")]
    )
    st.altair_chart((price_chart + fair_chart).properties(height=320).interactive(), width="stretch")

with tab2:
    st.subheader("EPS × 本益比合理價矩陣")
    eps_values = np.linspace(forward_eps * 0.8, forward_eps * 1.2, 5)
    pe_values = np.linspace(float(base_row["合理本益比"]) * 0.8, float(base_row["合理本益比"]) * 1.2, 5)
    matrix = pd.DataFrame([[eps * pe for pe in pe_values] for eps in eps_values], index=[f"EPS {v:,.0f}" for v in eps_values], columns=[f"P/E {v:.1f}x" for v in pe_values])
    st.dataframe(matrix.style.format("{:,.0f}"), width="stretch")
    st.caption("橫向是假設本益比，縱向是假設 EPS；可用來觀察獲利下修或估值壓縮時，合理價的變化幅度。")

with tab3:
    st.subheader("EV/EBITDA 企業價值拆解")
    ev_display = enterprise_valuation.copy()
    st.dataframe(
        ev_display,
        hide_index=True,
        width="stretch",
        column_config={
            "情境EBITDA/股": st.column_config.NumberColumn(format="%,.2f"),
            "合理EV/EBITDA": st.column_config.NumberColumn(format="%.1fx"),
            "每股企業價值": st.column_config.NumberColumn(format="%,.1f"),
            "淨負債/股": st.column_config.NumberColumn(format="%,.1f"),
            "合理價格": st.column_config.NumberColumn(format="%,.1f"),
            "相對現價%": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.latex(r"EV = EBITDA \times (EV/EBITDA)\qquad Equity\ Value = EV + Cash - Debt")
    st.write("此處採每股口徑，因此合理股價＝EBITDA／股 × 合理 EV/EBITDA－淨負債／股。若公司持有淨現金，淨負債／股請輸入負數。")

with tab4:
    st.markdown("""
    **四階段判定**

    - 需求動能上升、庫存動能下降：主動去庫存（復甦初期）
    - 需求動能上升、庫存動能上升：主動補庫存（擴張期）
    - 需求動能下降、庫存動能上升：被動補庫存（景氣後期）
    - 需求動能下降、庫存動能下降：被動去庫存（收縮期）

    **P/E 合理價**：未來 12 個月 EPS × 歷史合理 P/E × 循環係數 × 利率調整係數。

    **企業價值合理價**：未來 12 個月 EBITDA／股 × 合理 EV/EBITDA × 循環與利率係數－淨負債／股。

    **綜合合理價**：依使用者設定權重，合併 P/E 法與企業價值法。獲利穩定的成熟公司可提高 P/E 權重；資本密集、折舊高或財務結構差異大的公司，可提高 EV/EBITDA 權重。

    **重要限制**：庫存循環是景氣框架，不是精準擇時工具；單月資料可能受基期與季節性影響。指數 EPS、資料發布落後、匯率及產業結構也會影響結果。建議至少用三個月趨勢確認，並搭配訂單、PMI、信用利差與估值分位。
    """)

result = blended.assign(市場=selected_market, 循環階段=cycle.phase, 市場分數=market_score, 分析日期=date.today().isoformat())
st.download_button("下載本次分析結果", result.to_csv(index=False).encode("utf-8-sig"), f"{selected_market.split('｜')[0]}_庫存循環合理價.csv", "text/csv")

st.divider()
st.header("景氣循環與庫存循環總覽")
st.caption("兩組循環使用相同月資料與本頁假設；上傳實際歷史資料後會同步重算。景氣分數是需求水準與需求動能的模型指標，並非官方領先指標。")

with st.container(border=True):
    st.subheader("景氣循環 × 庫存循環對照圖")
    st.altair_chart(cycle_framework_chart(cycle.phase), width="stretch")
    st.caption(f"目前模型位於「{cycle.phase}」：{PHASES[cycle.phase]['quadrant']}，對應 {PHASES[cycle.phase]['signal']}。菱形標記為目前位置。")

business_history = business_cycle_history(history).dropna(subset=["DemandMomentum"]).copy()
business_colors = alt.Scale(
    domain=["復甦", "擴張", "放緩", "收縮"],
    range=["#29B6A6", "#3B82F6", "#F59E0B", "#EF4444"],
)

with st.container(border=True):
    st.subheader("景氣循環圖與近 12 期表")
    business_line = alt.Chart(business_history).mark_line(color="#94A3B8", strokeWidth=2).encode(
        x=alt.X("Date:T", title="日期"),
        y=alt.Y("BusinessScore:Q", title="景氣循環分數", scale=alt.Scale(domain=[0, 100])),
    )
    business_points = alt.Chart(business_history).mark_circle(size=80).encode(
        x=alt.X("Date:T", title="日期"),
        y=alt.Y("BusinessScore:Q", title="景氣循環分數", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("BusinessPhase:N", title="景氣階段", scale=business_colors),
        tooltip=[
            alt.Tooltip("Date:T", title="日期", format="%Y-%m"),
            alt.Tooltip("BusinessPhase:N", title="景氣階段"),
            alt.Tooltip("DemandGrowth:Q", title="需求年增率", format="+.2f"),
            alt.Tooltip("DemandMomentum:Q", title="需求動能", format="+.2f"),
            alt.Tooltip("BusinessScore:Q", title="景氣分數", format=".1f"),
        ],
    )
    st.altair_chart((business_line + business_points).properties(height=330).interactive(), width="stretch")
    business_table = business_history.tail(12)[["Date", "DemandGrowth", "DemandMomentum", "BusinessPhase", "BusinessScore"]].rename(columns={
        "Date": "日期", "DemandGrowth": "需求年增率%", "DemandMomentum": "需求動能變化", "BusinessPhase": "景氣階段", "BusinessScore": "景氣分數",
    })
    st.dataframe(
        business_table.sort_values("日期", ascending=False), hide_index=True, width="stretch",
        column_config={
            "日期": st.column_config.DateColumn(format="YYYY-MM"),
            "需求年增率%": st.column_config.NumberColumn(format="%+.2f%%"),
            "需求動能變化": st.column_config.NumberColumn(format="%+.2f"),
            "景氣分數": st.column_config.NumberColumn(format="%.1f"),
        },
    )

with st.container(border=True):
    st.subheader("庫存循環四象限圖與近 12 期表")
    inventory_history = history.dropna(subset=["DemandMomentum", "InventoryMomentum"]).copy().tail(24)
    inventory_history["最新一期"] = False
    if not inventory_history.empty:
        inventory_history.loc[inventory_history.index[-1], "最新一期"] = True
    zero_rules = pd.DataFrame({"zero": [0]})
    vertical_zero = alt.Chart(zero_rules).mark_rule(color="#64748B", strokeDash=[5, 5]).encode(x="zero:Q")
    horizontal_zero = alt.Chart(zero_rules).mark_rule(color="#64748B", strokeDash=[5, 5]).encode(y="zero:Q")
    quadrant_points = alt.Chart(inventory_history).mark_circle(opacity=0.85).encode(
        x=alt.X("InventoryMomentum:Q", title="庫存動能變化（百分點）"),
        y=alt.Y("DemandMomentum:Q", title="需求動能變化（百分點）"),
        color=alt.Color("Phase:N", title="庫存循環", scale=alt.Scale(domain=list(PHASES), range=[PHASES[p]["color"] for p in PHASES])),
        size=alt.Size("最新一期:N", legend=None, scale=alt.Scale(domain=[False, True], range=[65, 260])),
        tooltip=[
            alt.Tooltip("Date:T", title="日期", format="%Y-%m"),
            alt.Tooltip("Phase:N", title="循環階段"),
            alt.Tooltip("DemandMomentum:Q", title="需求動能", format="+.2f"),
            alt.Tooltip("InventoryMomentum:Q", title="庫存動能", format="+.2f"),
        ],
    )
    st.altair_chart((quadrant_points + vertical_zero + horizontal_zero).properties(height=390).interactive(), width="stretch")
    st.caption("右上＝主動補庫存、左上＝主動去庫存、右下＝被動補庫存、左下＝被動去庫存；較大的圓點是最新一期。")
    inventory_table = history.dropna(subset=["DemandMomentum", "InventoryMomentum"]).tail(12)[[
        "Date", "DemandGrowth", "InventoryGrowth", "DemandMomentum", "InventoryMomentum", "Phase"
    ]].rename(columns={
        "Date": "日期", "DemandGrowth": "需求年增率%", "InventoryGrowth": "庫存年增率%",
        "DemandMomentum": "需求動能變化", "InventoryMomentum": "庫存動能變化", "Phase": "庫存循環階段",
    })
    st.dataframe(
        inventory_table.sort_values("日期", ascending=False), hide_index=True, width="stretch",
        column_config={
            "日期": st.column_config.DateColumn(format="YYYY-MM"),
            "需求年增率%": st.column_config.NumberColumn(format="%+.2f%%"),
            "庫存年增率%": st.column_config.NumberColumn(format="%+.2f%%"),
            "需求動能變化": st.column_config.NumberColumn(format="%+.2f"),
            "庫存動能變化": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

st.caption("內建數字是示範假設；若要正式研究，請以上傳的公司／市場實際庫存、需求與 EPS 資料取代。")

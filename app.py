"""Week 12 HW — Part B: Stroop Dataset Hygiene & Caching Dashboard.

作者：Irene Ho（NS5116）

設計重點
--------
- **單一來源**：清理 / 分析邏輯一律來自 `pipeline.py`（與 report.ipynb 相同），
  dashboard 不另寫一份 clean()，避免 notebook 與 app 漂移。
- **Caching**：用 `@st.cache_data` 快取「讀檔」與「清理」兩個昂貴步驟；
  sidebar 即時顯示兩者用時，第一次 = cache miss（慢）、之後 = cache hit（<1ms）。
  改 sidebar 篩選**不會**觸發 re-clean，因為 clean() 的輸入（原始 df）沒變。
- **outlier_sd 是分析參數**：放在 sidebar → 傳給 `analyse()`，不混進 clean()
  （對應 report A.6 的 cleaning vs analysis 邊界）。
"""
from __future__ import annotations

import contextlib
import io
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pipeline import analyse, clean

DATA_PATH = Path("data") / "messy_stroop_homework.csv"
CONDS = ["congruent", "incongruent"]


# ---------------------------------------------------------------------------
# B.1 Cached loaders（兩個昂貴步驟各自快取）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="讀取原始 CSV...")
def load_data(path: str) -> pd.DataFrame:
    """讀原始 CSV。ttl=600：資料檔極少改動，10 分鐘 TTL 足以兼顧新鮮度與效能。"""
    return pd.read_csv(path)


@st.cache_data(show_spinner="清理資料中（呼叫 pipeline.clean）...")
def clean_cached(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """包一層 clean()：對相同輸入 deterministic，故可安全快取。

    順便擷取 clean() 的逐步 stdout log，回傳給 UI 在 expander 顯示
    （把 Part A 的「每步 row 變化」帶進 dashboard）。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cleaned = clean(raw)
    return cleaned, buf.getvalue()


def _timed(fn, *args, **kwargs):
    """回傳 (結果, 毫秒)。cache hit 時毫秒會趨近 0，用來示範快取效益。"""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000


def _analyse_safe(df: pd.DataFrame, outlier_sd: float):
    """analyse() 在缺少某一 condition 時會 KeyError；這裡攔截並降級。"""
    if not set(CONDS).issubset(df["condition"].unique()):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # 吃掉 analyse 的 print
            return analyse(df, outlier_sd=outlier_sd)
    except (KeyError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Stroop Hygiene Dashboard", layout="wide")
    st.title("🧹 Stroop Dataset Hygiene & Caching Dashboard")
    st.caption("Week 12 HW — Irene Ho｜清理 (Part A `clean()`) → 互動篩選 → "
               "分析 (`analyse()`)，全程示範 Streamlit caching")

    # ---- Load + clean（計時以示範快取效益）----
    raw, t_load = _timed(load_data, str(DATA_PATH))
    (df, clean_log), t_clean = _timed(clean_cached, raw)

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚡ Caching")
        st.metric("load_data 用時", f"{t_load:.1f} ms")
        st.metric("clean() 用時", f"{t_clean:.1f} ms")
        st.caption(
            "首次執行 = **cache miss**（慢）；之後同樣輸入 = **cache hit**（<1 ms）。"
            "調整下方篩選**不會**重跑 clean()，因為它的輸入（原始 df）未變。"
        )
        if st.button("🗑 Clear cache（重現 cache miss）", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()

        st.header("🔎 篩選 (B.2)")
        subj_all = sorted(df["subject_id"].unique())
        subjects = st.multiselect("Subject ID", subj_all, default=subj_all)
        # floor/ceil：避免 int() 截斷把最快/最慢 trial 排除在預設範圍外
        rt_lo = math.floor(df["rt_ms"].min())
        rt_hi = math.ceil(df["rt_ms"].max())
        rt_min, rt_max = st.slider("RT 範圍 (ms)", rt_lo, rt_hi, (rt_lo, rt_hi))
        st.divider()

        st.header("📐 分析參數")
        outlier_sd = st.slider(
            "Outlier 修剪 mean ± k·SD", 1.5, 4.0, 3.0, 0.5,
            help="這是**分析決策**（不是 cleaning）：傳給 analyse()，不進 clean()。"
            "見 report A.6。",
        )

    # ---- Cleaning log（把 Part A 的逐步 log 帶進 dashboard）----
    with st.expander("🧹 Cleaning log — `clean()` 每步 row 變化（Part A）"):
        st.code(clean_log or "(cache hit：log 已於首次清理輸出到 terminal，"
                "按上方 Clear cache 可重現)", language="text")

    # ---- 套用篩選 + 邊界防呆 ----
    if not subjects:
        st.warning("請至少選擇一個 Subject ID。")
        st.stop()
    mask = df["subject_id"].isin(subjects) & df["rt_ms"].between(rt_min, rt_max)
    view = df[mask].copy()
    if view.empty:
        st.warning("目前篩選條件下沒有任何 trial，請放寬條件。")
        st.stop()

    res = _analyse_safe(view, outlier_sd)

    # ---- B.3 KPI ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n trials（篩選後）", f"{len(view)}")
    if res is not None:
        bc = res["by_condition"]
        c2.metric("Mean RT congruent", f"{bc.loc['congruent', 'mean']:.0f} ms")
        c3.metric("Mean RT incongruent", f"{bc.loc['incongruent', 'mean']:.0f} ms")
        c4.metric("Stroop effect", f"{res['stroop_effect_ms']:.1f} ms",
                  help=f"incongruent − congruent（已做 {outlier_sd} SD outlier 修剪，"
                       f"n={res['n_used']}）")
    else:
        for col in (c2, c3, c4):
            col.metric("—", "N/A")
        st.info("篩選後缺少某一 condition（congruent / incongruent），"
                "無法計算 Stroop effect，請放寬篩選。")

    # ---- B.3 圖表 ----
    left, right = st.columns(2)
    with left:
        st.subheader("Stroop effect（condition × mean RT）")
        if res is not None:
            bc = res["by_condition"]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(bc.index, bc["mean"], yerr=bc["std"], capsize=6,
                   color=["#4C72B0", "#C44E52"])
            ax.set_ylabel("Mean RT (ms)")
            ax.set_title(f"Stroop = {res['stroop_effect_ms']:.1f} ms "
                         f"({outlier_sd} SD trim, n={res['n_used']})")
            st.pyplot(fig)
        else:
            st.empty()

    with right:
        st.subheader("Mean RT by subject × condition")
        pivot = (view.groupby(["subject_id", "condition"])["rt_ms"]
                 .mean().unstack())
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        pivot.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Mean RT (ms)")
        ax2.set_xlabel("subject_id")
        st.pyplot(fig2)

    st.subheader("RT 分佈（清理後，依 condition）")
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    for cond in CONDS:
        sub = view.loc[view["condition"] == cond, "rt_ms"]
        if not sub.empty:
            ax3.hist(sub, bins=30, alpha=0.5, label=f"{cond} (n={len(sub)})")
    ax3.set_xlabel("rt_ms (ms)")
    ax3.legend()
    st.pyplot(fig3)

    # ---- B.3 資料表 ----
    st.subheader("Cleaned data preview（前 20 列）")
    st.dataframe(view.head(20), width="stretch")


if __name__ == "__main__":
    main()

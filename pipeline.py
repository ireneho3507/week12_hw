"""Single-source cleaning + analysis logic for the Week 12 HW.

`report.ipynb`（分析）與 `app.py`（dashboard）都 import 這裡的 `clean()` /
`analyse()`，確保 notebook 與 dashboard 用的是**同一套**邏輯，不會漂移。

每個 cleaning step 都對應回 notebook A.0 的 schema 表與 A.3 的觀察；
outlier 修剪刻意留在 `analyse()`（分析參數），不放進 `clean()`（資料清理）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """把 A.3 觀察到的每個資料品質問題修掉，回傳乾淨 DataFrame。

    每一步皆標明【觀察(對應 A.3 哪條 Obs) / 動作(用的 pandas API) / 代價(犧牲了什麼)】，
    並 print 該步的 row 數變化 before → after。

    需求合規：
      - Pure function：第一行 df = df.copy()，全程不改動呼叫端傳入的 df。
      - 不使用裸 df.dropna()；無效值一律用「明確布林遮罩」剔除（看得到自己 drop 什麼）。
      - outlier (mean ± SD) 修剪**不**在此（屬分析決策），留給 A.6 analyse()。
    """
    df = df.copy()  # 需求3：pure function — 之後所有操作都只作用在這份 copy 上

    # === Step 1：移除整列重複 ===
    # 觀察：對應 A.3 Obs 3.5 — df.duplicated() = 3（整列複製，生成器第 108–110 行）
    # 動作：DataFrame.drop_duplicates()（全欄比對，非只看 trial_id）
    # 代價：損失 3 row；若存在兩筆內容完全相同的「合法」trial 也會被併掉（本資料不會）
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"step1 drop_duplicates      : {before} → {len(df)}  (lost {before - len(df)} rows)")

    # === Step 2：condition 6 raw levels → 2 ===
    # 觀察：對應 A.3 Obs 3.3 — 尾端空白 / 大小寫 / 縮寫共 6 個 raw level
    # 動作：Series.str.strip().str.lower() + Series.replace() 把縮寫 map 回標準字
    # 代價：假設 con==congruent、incong.==incongruent（依生成器第 90–91 行）；轉換不可逆
    before = len(df)
    lv_before = sorted(df["condition"].unique().tolist())
    df["condition"] = df["condition"].str.strip().str.lower()
    df["condition"] = df["condition"].replace({"con": "congruent",
                                               "incong.": "incongruent"})
    bad = ~df["condition"].isin(["congruent", "incongruent"])
    if bad.any():
        raise ValueError(f"未預期的 condition: {df.loc[bad, 'condition'].unique()}")
    print(f"step2 condition normalize  : {before} → {len(df)}  (lost {before - len(df)} rows)"
          f"  | levels {lv_before} → {sorted(df['condition'].unique().tolist())}")

    # === Step 3a：rt_ms 轉數值 + 清 sentinel ===
    # 觀察：對應 A.3 Obs 3.1（字串 "missing"/"--"）+ Obs 3.2（數值 -1/9999）
    # 動作：pd.to_numeric(errors="coerce") 讓字串變 NaN；Series.replace 把 -1/9999 換 NaN；
    #       再用布林遮罩 df[df["rt_ms"].notna()] 留下有效值（需求4：不用裸 dropna()）
    # 代價：無效 rt 的 trial 無法分析 RT，必須整列剔除（≈ 9% 資料）
    before = len(df)
    df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    df["rt_ms"] = df["rt_ms"].replace({-1: np.nan, 9999: np.nan})
    df = df[df["rt_ms"].notna()].copy()
    print(f"step3a rt_ms 去無效值       : {before} → {len(df)}  "
          f"(lost {before - len(df)} rows：字串+數值 sentinel)")

    # === Step 3b：rt_ms 限定 200–2500 ms ===
    # 觀察：對應 A.3 Obs 3.2 / A.0 schema — 文獻 Whelan (2008) cutoff
    # 動作：Series.between(200, 2500) 布林遮罩
    # 代價：排除 anticipation(<200)/lapse(>2500)；本資料 sentinel 已清，預期攔 0（護欄）
    before = len(df)
    df = df[df["rt_ms"].between(200, 2500)].copy()
    print(f"step3b rt_ms 範圍 200–2500  : {before} → {len(df)}  "
          f"(lost {before - len(df)} rows：本資料應為 0)")

    # === Step 4：accuracy 字串編碼 → int ===
    # 觀察：對應 A.3 Obs 3.4 — 混入字串 "True"/"False"
    # 動作：Series.replace({"True":1,"False":0}) → pd.to_numeric → astype(int)
    # 代價：假設 "True"==正確（依生成器第 100 行）；轉 int 後遺失原始字串編碼資訊
    before = len(df)
    df["accuracy"] = df["accuracy"].replace({"True": 1, "False": 0})
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="raise").astype(int)
    print(f"step4 accuracy str→int     : {before} → {len(df)}  (lost {before - len(df)} rows)"
          f"  | dtype={df['accuracy'].dtype}, unique={sorted(df['accuracy'].unique())}")

    # === Step 5：age sentinel → NaN（刻意不 drop row）===
    # 觀察：對應 A.3 Obs 3.5 — age 含 -1 / 888 / NaN
    # 動作：Series.replace({-1:np.nan, 888:np.nan})；刻意不 drop（需求4：更不用裸 dropna）
    # 代價：age 缺值率上升；但本作業分析不依賴 age，drop 反而會無謂損失 rt 有效的 trial
    before = len(df)
    n_bad = int(df["age"].isin([-1, 888]).sum())
    df["age"] = df["age"].replace({-1: np.nan, 888: np.nan})
    print(f"step5 age sentinel→NaN     : {before} → {len(df)}  (lost {before - len(df)} rows)"
          f"  | 換掉 {n_bad} 個 (-1/888)，age NaN 共 {int(df['age'].isna().sum())}（刻意保留 row）")

    return df.reset_index(drop=True)


def analyse(df: pd.DataFrame, *, outlier_sd: float = 3.0) -> dict:
    """計算 Stroop effect。outlier_sd 為**分析參數**（可調），不是 cleaning 規則。"""
    # Distribution-based trim（Whelan 2008）：mean ± outlier_sd × SD
    mu, sd = df["rt_ms"].mean(), df["rt_ms"].std()
    mask = (df["rt_ms"] - mu).abs() <= outlier_sd * sd
    df_trim = df[mask]
    print(f"outlier trim ({outlier_sd} SD): {len(df)} -> {len(df_trim)} rows "
          f"(剔除 {len(df) - len(df_trim)})")

    by_cond = df_trim.groupby("condition")["rt_ms"].agg(["mean", "std", "count"])
    cong = by_cond.loc["congruent", "mean"]
    incong = by_cond.loc["incongruent", "mean"]
    stroop_effect = incong - cong
    return {
        "by_condition": by_cond,
        "stroop_effect_ms": stroop_effect,
        "n_used": len(df_trim),
    }

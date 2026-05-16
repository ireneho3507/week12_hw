# Week 12 HW — Dataset Hygiene & Caching Dashboard

> NS5116 電腦硬體與程式語言在行為科學實驗與大數據分析之應用
> 作者：Irene Ho（中央大學認知神經科學研究所）

## 1. 怎麼跑

```bash
# 安裝套件
pip install -r requirements.txt

# 執行 Streamlit dashboard
python -m streamlit run app.py
# 開啟 http://localhost:8501
```

Notebook：用 Jupyter 開啟 `report.ipynb`。

## 2. 資料來源

- Dataset：`./data/messy_stroop_homework.csv`（n = 243 rows = 240 base + 3 注入重複，由 `generate_messy_stroop_homework.py` 生成，seed=2026）
- 文獻：
  - Whelan, R. (2008). Effective analysis of reaction time data. *The Psychological Record*, 58(3), 475–482.
  - MacLeod, C. M. (1991). Half a century of research on the Stroop effect: An integrative review. *Psychological Bulletin*, 109(2), 163–203.

## 3. 三條最重要的 cleaning 決定

1. **`rt_ms` 採文獻嚴格範圍 200–2500 ms**（依據：文獻 Whelan 2008 + 觀察）：先 `to_numeric(coerce)` 把字串 sentinel `"missing"`/`"--"` 變 NaN，再把數值 sentinel `-1`/`9999` 換 NaN，最後保留 200–2500 ms；無效 rt 的 trial 無法分析 RT，必須剔除（243→219）。
2. **`condition` 6 raw levels 正規化為 2**（依據：生成器第 41/90–91 行 + 觀察）：`strip().lower()` 後把縮寫 `con`→`congruent`、`incong.`→`incongruent`，並驗證最終只剩兩個 level。
3. **整列重複用 `drop_duplicates()` 全欄比對而非只看 `trial_id`**（依據：生成器第 108–110 行是整列複製）：避免誤刪合法 trial，移除 3 row。

> `accuracy` 字串 `"True"`/`"False"` → `1`/`0` 後轉 int；`age` 的 `-1`/`888` 換 NaN 但**不 drop row**（分析不依賴 age，保留資料量）。

## 4. `outlier_sd` 為什麼放在 `analyse()` 而不是 `clean()`

`clean()` 修的是**客觀資料錯誤**（型別、sentinel、拼字、重複），結果唯一且不帶研究判斷。mean ± k·SD 的 outlier 修剪則是**分析決策**——用 2 還是 3 SD 是主觀選擇，會直接改變結論（本資料 `outlier_sd=2.0` → Stroop ≈ 63 ms、`3.0` → ≈ 77 ms）。把它做成 `analyse(outlier_sd=3.0)` 的具名參數，TA 可一眼看到並做敏感度分析，`clean()` 維持純粹 deterministic。最終 Stroop effect = **77.1 ms**（3 SD），與生成器注入的 +80 ms 一致，落於 MacLeod (1991) 的 50–100 ms 範圍。

## 5. 部署

- GitHub repo：https://github.com/ireneho3507/week12_hw
- Streamlit Cloud：https://week12hw-irene-stroop.streamlit.app/

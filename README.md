# Ag2Se CORE / CORE-SHELL Spectrum Prediction

本專案以已知條件的 Ag2Se CORE 與 CORE-SHELL 光譜，預測指定濃度比或 pH 條件下的完整光譜。

需要 Python 3.10 或更新版本。

## 模型設計

- 以 leave-one-spectrum-out cross-validation 選擇振幅、峰位、FWHM 策略與 XGBoost 超參數。
- 結合光譜前處理、physics-informed prior 與 XGBoost residual learning。
- 支援 normalized 與 raw-scale 光譜輸出，以及獨立的模型評估流程。

## 方法

1. 對訓練光譜做 median filter、Savitzky-Golay smoothing 與共同波長網格插值。
2. 僅由訓練資料外推 peak position、amplitude、baseline 與 FWHM。
3. 建立 quality-weighted、peak-aligned physics prior。
4. 以 XGBoost 學習 leave-one-out prior residual；只有訓練端 CV 優於 prior 時才啟用 residual model。
5. 套用由訓練端 CV 決定的 FWHM 處理，輸出 normalized 與 raw-scale 預測。

LHS 用於交叉驗證超參數搜尋；若要快速執行，可設 `--lhs-samples 0` 使用固定參數。

## 安裝

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 資料夾格式

資料不包含在 repository 中。`--data-root` 應指向原始「訓練資料集」資料夾；程式預期以下四組子資料夾：

```text
<data-root>/
├─ 只有核/
│  ├─ Ag2Se_濃度比(PL)/
│  └─ Ag2Se_pH值(6~11pH)/
└─ 有核也有殼/
   ├─ CS_不同濃度比(PL)/
   └─ CS_pH值(7-11 pH)/
```

每個文字檔為兩個數值欄位：wavelength 與 intensity。

## 執行

四組一起執行：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset all
```

只跑一組，並使用 GPU：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset PH-SHELL --device cuda
```

只輸出預測結果：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset PH-SHELL --no-evaluate
```

產物寫入 `outputs/<dataset>/`，該目錄已由 `.gitignore` 排除。GitHub 只需上傳本目錄內的程式碼與說明檔，不需上傳 data、圖片或執行結果。

## 測試

```bash
pip install -r requirements-dev.txt
pytest -q
```

測試涵蓋主要訓練與預測介面。

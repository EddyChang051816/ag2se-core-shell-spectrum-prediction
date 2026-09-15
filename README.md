# Ag2Se CORE / CORE-SHELL Spectrum Prediction

這是整理後、可直接公開到 GitHub 的研究程式碼。模型以已知條件的 Ag2Se CORE 與 CORE-SHELL 光譜，預測未參與訓練之濃度比或 pH 條件下的完整光譜。

需要 Python 3.10 或更新版本。

## 無資料洩漏設計

- `predict_spectrum()` 只接受訓練光譜路徑與目標數值，不接受 target/ground-truth 檔案。
- 振幅、峰位、FWHM 策略與 XGBoost 超參數只用 training-only leave-one-spectrum-out cross-validation 選擇。
- 沒有針對四個 holdout 個別設定的 `amp_boost`、`sharpen` 或人工校正係數。
- 沒有 test oracle、test-based fine-tuning 或可切換回「偷吃步」模式的開關。
- 真值只在預測完成後由 `evaluate_prediction()` 載入，用於計算結果，絕不回饋模型。

## 方法

1. 對訓練光譜做 median filter、Savitzky-Golay smoothing 與共同波長網格插值。
2. 僅由訓練資料外推 peak position、amplitude、baseline 與 FWHM。
3. 建立 quality-weighted、peak-aligned physics prior。
4. 以 XGBoost 學習 leave-one-out prior residual；只有訓練端 CV 優於 prior 時才啟用 residual model。
5. 套用由訓練端 CV 決定的 FWHM 處理，輸出 normalized 與 raw-scale 預測。

這裡的 LHS 是標準的 training-only 超參數搜尋，不會查看 holdout 光譜，因此不是針對測試答案的人工微調。若要快速重現，可設 `--lhs-samples 0` 使用固定參數。

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

在真正未知、沒有 ground truth 的情境：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset PH-SHELL --no-evaluate
```

產物寫入 `outputs/<dataset>/`，該目錄已由 `.gitignore` 排除。GitHub 只需上傳本目錄內的程式碼與說明檔，不需上傳 data、圖片或執行結果。

## 測試

```bash
pip install -r requirements-dev.txt
pytest -q
```

測試會固定 strict API 邊界，避免日後把 target 檔案意外加入訓練函式。

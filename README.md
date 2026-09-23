# Ag2Se 核心／核殼光譜預測

本專案利用已知條件下的 Ag2Se 核心（CORE）與核殼（CORE-SHELL）光譜，預測指定濃度比或 pH 條件下的完整光譜。

需要 Python 3.10 或更新版本。

## 模型設計

- 使用留一光譜交叉驗證（leave-one-spectrum-out cross-validation），選擇振幅、峰位、半高全寬（FWHM）策略與 XGBoost 超參數。
- 結合光譜前處理、物理資訊先驗模型與 XGBoost 殘差學習。
- 支援正規化與原始尺度的光譜輸出，並提供獨立的模型評估流程。

## 方法

1. 對訓練光譜進行中值濾波、Savitzky-Golay 平滑處理，以及共同波長網格插值。
2. 僅使用訓練資料推估峰位、振幅、基線與半高全寬。
3. 建立依品質加權並對齊峰位的物理資訊先驗模型。
4. 使用 XGBoost 學習留一法先驗殘差；只有當訓練端交叉驗證結果優於先驗模型時，才啟用殘差模型。
5. 套用由訓練端交叉驗證決定的半高全寬處理方式，輸出正規化與原始尺度的預測結果。

拉丁超立方抽樣（LHS）用於交叉驗證的超參數搜尋；若要快速執行，可設定 `--lhs-samples 0` 使用固定參數。

## 安裝

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 資料夾格式

本程式碼倉庫不包含資料。`--data-root` 應指向原始的「訓練資料集」資料夾；程式預期其中包含以下四組子資料夾：

```text
<data-root>/
├─ 只有核/
│  ├─ Ag2Se_濃度比(PL)/
│  └─ Ag2Se_pH值(6~11pH)/
└─ 有核也有殼/
   ├─ CS_不同濃度比(PL)/
   └─ CS_pH值(7-11 pH)/
```

每個文字檔應包含兩個數值欄位：波長（wavelength）與強度（intensity）。

## 執行

一次執行全部四組資料：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset all
```

只執行一組資料，並使用 GPU：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset PH-SHELL --device cuda
```

只輸出預測結果：

```bash
python train.py --data-root "C:\path\to\訓練資料集" --dataset PH-SHELL --no-evaluate
```

產出檔案會寫入 `outputs/<dataset>/`，該目錄已由 `.gitignore` 排除。GitHub 只需上傳本目錄內的程式碼與說明文件，不需上傳資料、圖片或執行結果。

## 測試

```bash
pip install -r requirements-dev.txt
pytest -q
```

測試涵蓋主要的訓練與預測介面。

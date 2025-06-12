
# 🌸 Flower Image Classification using CNN (TensorFlow)

本專案使用 TensorFlow 搭建卷積神經網路（CNN），針對 [flower_photos](https://www.tensorflow.org/tutorials/load_data/images) 資料集中五類花卉（daisy、dandelion、roses、sunflowers、tulips）進行影像分類任務。最終模型在測試集上達到 **79.5% 準確率**，其中向日葵（sunflowers）分類效果最佳。

---

## 📁 資料集簡介

資料集共包含五類花卉，共 3,670 張圖片。依照 7:2:1 比例劃分為訓練集、驗證集與測試集，並確保每類比例分佈均衡。

| 類別       | 總數 | 訓練集 | 驗證集 | 測試集 |
|------------|------|--------|--------|--------|
| daisy      | 633  | 443    | 127    | 63     |
| dandelion  | 898  | 629    | 180    | 89     |
| roses      | 641  | 449    | 128    | 64     |
| sunflowers | 699  | 489    | 140    | 70     |
| tulips     | 799  | 559    | 160    | 80     |

---

## 🧠 模型架構

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(180,180,3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
```

- **Optimizer**：Adam (lr=0.001)
- **Loss Function**：Sparse Categorical Crossentropy
- **Data Augmentation**：
  - 隨機水平翻轉
  - ±10°旋轉
  - ±10%縮放

---

## 📊 訓練成果

| 指標     | 訓練集 | 驗證集 | 測試集 |
|----------|--------|--------|--------|
| Accuracy | 84.7%  | 75.3%  | **79.5%** |

### 📌 測試集分類 F1-score：

| 類別       | F1-score |
|------------|----------|
| daisy      | 0.794    |
| dandelion  | 0.787    |
| roses      | 0.685    |
| sunflowers | **0.867** |
| tulips     | 0.753    |

---

## 🖼️ 成果視覺化

> 📌 建議你上傳以下圖檔並放在 `./assets/` 資料夾中再補上圖片連結

| 訓練準確率曲線 | 損失函數曲線 |
|----------------|----------------|
| ![acc](assets/accuracy_plot.png) | ![loss](assets/loss_curve.png) |

---

## 🚀 使用方式

```bash
# 安裝所需套件
pip install -r requirements.txt

# 執行訓練程式
python train.py
```

---

## 🔧 優化建議（已實施或待進行）

- ✅ 採用數據增強提升泛化能力
- ✅ 添加 Dropout 抑制過擬合
- ⏳ 準備引入 EfficientNet / ResNet 進行遷移學習
- ⏳ 強化影像增強（如 Cutout, Color Jitter）
- ⏳ 嘗試動態學習率調整策略（Cosine Decay）




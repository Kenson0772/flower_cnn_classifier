import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # 關掉 TF 的雜訊
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 字體設定
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 參數 ==========
DATA_DIR = r"C:\Users\kenso\Downloads\flower_photos"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123

# ========== 讀取資料 ========== 
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

valtest_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_batches = int(0.67 * len(valtest_ds_raw))
val_ds_raw  = valtest_ds_raw.take(val_batches)
test_ds_raw = valtest_ds_raw.skip(val_batches)

class_names = train_ds_raw.class_names
print("類別：", class_names)

# ========== 資料前處理 ==========
norm = tf.keras.layers.Rescaling(1./255)
aug  = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

def prep(ds, train=False):
    ds = ds.map(lambda x,y: (norm(x), y))
    if train:
        ds = ds.map(lambda x,y: (aug(x, training=True), y))
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = prep(train_ds_raw,  train=True)
val_ds   = prep(val_ds_raw)
test_ds  = prep(test_ds_raw)

# ========== 建立模型 ==========
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ========== 訓練模型 ==========
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop],
    verbose=1
)

# ========== 準確率評估 ==========
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"訓練準確率：{train_acc:.3f}")
print(f"驗證準確率：{val_acc:.3f}")
print(f"測試準確率：{test_acc:.3f}")

# ========== 訓練曲線 ==========
plt.figure(figsize=(5,4))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()

# ========== 損失曲線 ==========
plt.figure(figsize=(5,4))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()

# ========== 混淆矩陣 ==========
y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(p.argmax(axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Pred"); plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n分類報告：")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# ========== 測試圖 ==========
right_imgs, right_true, right_pred = [], [], []
wrong_imgs, wrong_true, wrong_pred = [], [], []

for imgs, labels in test_ds:                  # 走訪完整測試集
    probs = model.predict(imgs, verbose=0)
    preds = probs.argmax(axis=1)
    for img, t, p in zip(imgs.numpy(), labels.numpy(), preds):
        if t == p and len(right_imgs) < 5:
            right_imgs.append(img); right_true.append(t); right_pred.append(p)
        elif t != p and len(wrong_imgs) < 5:
            wrong_imgs.append(img); wrong_true.append(t); wrong_pred.append(p)
    if len(right_imgs) == 5 and len(wrong_imgs) == 5:
        break

def plot_two_rows(right_imgs, wrong_imgs,
                  right_true, right_pred,
                  wrong_true, wrong_pred,
                  fname="examples_10.png"):
    plt.figure(figsize=(15,6))
    # 上排：正確
    for i,(img,t,p) in enumerate(zip(right_imgs,right_true,right_pred)):
        plt.subplot(2,5,i+1)
        plt.imshow(img); plt.axis('off')
        plt.title(f"T:{class_names[t]}\nP:{class_names[p]}", color='green', fontsize=9)
    # 下排：誤判
    for j,(img,t,p) in enumerate(zip(wrong_imgs,wrong_true,wrong_pred)):
        plt.subplot(2,5,5+j+1)
        plt.imshow(img); plt.axis('off')
        plt.title(f"T:{class_names[t]}\nP:{class_names[p]}", color='red', fontsize=9)

    plt.suptitle("正確案例與誤判案例各 5 張", fontsize=16)
    plt.tight_layout(); plt.subplots_adjust(top=0.85)
    plt.savefig(fname, dpi=300)
    plt.show()

plot_two_rows(right_imgs, wrong_imgs,
              right_true, right_pred,
              wrong_true, wrong_pred)

# ========== 模型儲存 ==========
model.save("flower_cnn_final.keras")            # Keras 格式
model.export("flower_cnn_savedmodel")           # TensorFlow SavedModel
print("模型已存檔")

# 專案實戰解析：基於深度學習建構卷積神經網路模型演算法，實現圖像辨識分類

(https://colab.research.google.com/drive/1Fb4Lrn8sovB8999nKf4RMP7hejy_XArb?usp=drive_link)

## 📋 目錄

- [前言]
- [基礎知識介紹]
- [數據集收集]
- [模型訓練]
- [圖像辨識分類]
- [結果展示]
- [總結]
  
---

## 前言

隨著人工智慧的不斷發展，深度學習這門技術也越來越重要，許多人開啟了學習機器學習。本專案透過實戰案例，帶領大家從零開始設計實現一款基於深度學習的圖像辨識演算法。

**學習本計畫內容，你需要掌握以下基礎知識：**

1. Python 基礎語法
2. 計算機視覺庫 (OpenCV)
3. 深度學習框架 (TensorFlow)
4. 卷積神經網路 (CNN)

---

## 基礎知識介紹

### 1. Python
Python 是一個高層次的結合了解釋性、編譯性、互動性和物件導向的腳本語言。
- 學習連結：[Python學習](https://www.runoob.com/python3/python3-intro.html)

### 2. OpenCV
OpenCV 是一個開源的跨平台計算機視覺庫，實現了圖像處理和計算機視覺方面的許多通用演算法。
- 學習連結：[OpenCV學習](https://docs.opencv.org/4.x/index.html)

### 3. TensorFlow
TensorFlow 是Google開源的計算框架，可以很好地支援深度學習的各種演算法。
- 學習連結：[TensorFlow學習](https://tensorflow.google.cn/)

### 4. CNN (卷積神經網路)
卷積神經網路是一類包含卷積計算且具有深度結構的前饋神經網路，是深度學習的代表性演算法之一。
- 學習連結：[CNN學習](https://xie.infoq.cn/article/c4d846096c92c7dfcd6539075)

---

## 數據集收集
在進行影像辨識前，首先需要收集資料集，其次對於資料集做預處理，然後才能通過
深度卷積神經網路來進行特徵學習，得到估計分類模型。對於資料集的要求，在卷積神經網絡
（CNN）中，由於對輸入影像向量的權值參數的數量是固定的，所以在用卷積網路（CNN）對資料集
進行模型訓練前需進行影像預處理，確保輸入的影像尺寸是固定一致的。
<img width="536" height="45" alt="JP9DRjfC7CRtFiN8c_TjfLoWYoezGqym0WUiWeriqoexe8JmGJ2qV11IIg1feOiOAcXYWms7wVndFAjUeKFSIjvPZsT-puVcrBImOUMp6Ua_n8kNK5qX-PLfvhDA4YLLd39m5bddEUK5IcImQSg8Tw_3vP8E7dxxJN47LkDmsIF1WNKNT31djIeRbUbc1_K5tEp3vJrTztXX1AuBzfBileR" src="https://github.com/user-attachments/assets/eda0a3d3-0177-407f-871c-5431b6ab403a" />
圖一 分類網路模型流程圖

# 程式

**安裝套件**

<img width="628" height="185" alt="螢幕擷取畫面 2025-12-28 205745" src="https://github.com/user-attachments/assets/efa11ff3-85db-4e25-ab7c-d0c57cdc0b6f" />

---

**初始化與屬性設定**

<img width="657" height="237" alt="image" src="https://github.com/user-attachments/assets/2d9d247c-a798-4947-9dec-0a8604d14cce" />

---

**圖片處理相關**

<img width="695" height="508" alt="image" src="https://github.com/user-attachments/assets/162a7f09-703f-4272-9141-c7ef724c3e3e" />

---

**模型建立與訓練**

<img width="580" height="619" alt="image" src="https://github.com/user-attachments/assets/c0f654a3-b76d-47d0-b8ee-2ac129da74ab" />

<img width="727" height="594" alt="image" src="https://github.com/user-attachments/assets/e49fb419-1e3d-4b5b-b76a-75e23330a63f" />

---

**預測與結果顯示**

<img width="600" height="727" alt="image" src="https://github.com/user-attachments/assets/c6f9be5c-7e13-442d-af7b-71c97c975b9a" />

---

<img width="453" height="155" alt="image" src="https://github.com/user-attachments/assets/f75ebcc2-5083-4524-ad92-3b4794dc4fc9" />


**建立分類器:**

        classifier = SafeImageClassifier(image_path)

image_path 是資料集根目錄。會自動掃描資料夾，將每個子資料夾視為一個類別。初始化時會列出類別名稱。

**圖片處理:**

        classifier.resize_images()
        
將每個類別資料夾中的圖片 resize 成固定大小（200×200）。

避免訓練時因為尺寸不同造成錯誤。
        
        classifier.generate_csv()

生成 CSV，記錄每張圖片路徑與對應標籤。

CSV 方便後續讀取訓練資料。

---

**建立模型:**

        classifier.build_model()

建立 CNN 模型：

        3 層卷積 + 最大池化 + BatchNorm、 GlobalAveragePooling、 Dense + Dropout、 最後輸出類別數的 softmax

編譯模型，設定損失函數與優化器。

---

**訓練模型**

        classifier.train(epochs=10, batch_size=2)

讀取 CSV 中的圖片與標籤。

將標籤 one-hot encoding。

切分訓練集與驗證集（80%/20%）。

開始訓練模型。

---

# 結果圖

<img width="580" height="612" alt="image" src="https://github.com/user-attachments/assets/4accef56-40ed-4388-83a3-9a09cb2c5e25" />

<img width="559" height="597" alt="image" src="https://github.com/user-attachments/assets/c3985ebb-d758-4cc7-be0c-446fc1361250" />

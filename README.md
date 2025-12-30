# 專案實戰解析：基於深度學習建構卷積神經網路模型演算法，實現圖像辨識分類

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AMu-NHA90gZi46nU30Hw_G_bmK2yZ7zC?usp=sharing)

## 📋 目錄

- [前言](#前言)
- [基礎知識介紹](#基礎知識介紹)
- [數據集收集](#數據集收集)
- [模型訓練](#模型訓練)
- [圖像辨識分類](#圖像辨識分類)
- [結果展示](#結果展示)
- [總結](#總結)
  
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

本案例以實現**垃圾分類識別**作為最終目標，數據集包含四類圖片：

- 廚餘垃圾 (Kitchen waste)
- 可回收垃圾 (Recyclable)
- 有毒垃圾 (Hazardous)
- 其它垃圾 (Other)

每類圖片數據集規模為200張（學習者可依需求選擇數據集類型及規模）。

![Flow Chart](images/flowchart.png)
*圖一：分類網路模型流程圖* 

![Directories](images/directories.png)  
*圖二：數據集目錄結構*  

### 數據預處理流程

#### 1. 圖片重命名
```python
#數據圖片rename
#數據集路徑：(self.image_path = "./picture/")
   def rename(self):
        listdir = os.listdir(self.image_path)
        i = 0
        while i < len(listdir):
            images_list_dir = os.listdir(os.path.join(self.image_path, listdir[i]))
            j = 0
            while j < len(images_list_dir):
                old_name = os.path.join(self.image_path, listdir[i], images_list_dir[j])
                new_name = os.path.join(self.image_path, "%d-%d" % (i, j) + ".jpg")
                os.rename(old_name, new_name)
                j += 1
            i += 1
        for p in range(len(listdir)):
            tmp_path = os.path.join(self.image_path, listdir[p])
            if os.path.exists(tmp_path):
                os.removedirs(tmp_path)
```

#### 2. 圖片尺寸统一
```python
#圖片resize
 def resize_img(self):
        listdir = os.listdir(self.image_path)
        for file in listdir:
            file_path = os.path.join(self.image_path, file)
            try:
                imread = cv2.imread(file_path)
                resize = cv2.resize(imread, (200, 200))
                cv2.imwrite(os.path.join(self.image_path, file), resize)
            except Exception:
                os.remove(file_path)
                continue

```

![After resize](images/after.png)  
*圖三：預處理後數據集範例*  

#### 3. 數據轉存為CSV
```python
#轉存圖片信息到csv文件
#csv生成路徑：(csv_file_saved_path = "./picture/")
def train_data_to_csv(self):
        files = os.listdir(self.image_path)
        data = []
        for file in files:
            data.append({"path": self.image_path + file, "label": file[0]})

        frame = pd.DataFrame(data, columns=['path', 'label'])
        dummies = pd.get_dummies(frame['label'], 'label')
        concat = pd.concat([frame, dummies], 1)
        concat.to_csv(csv_file_saved_path + "train.csv")

```

![CSV DEMO](images/csv.png)  
*圖四：數據集轉存CSV示例*

---

## 模型訓練

### 網路結構設計

本項目採用深度卷積神經網路，包含以下層次：

1. **卷積層1-4** (Conv Layer)
   - 卷積層：特徵提取
   - 池化層：降維
   - 批歸一化：加速收斂
   - Dropout：防止過擬合

2. **全連接層1-5** (FC Layer)
   - 逐步降維：256 → 128 → 64 → 32 → 5
   - 最終輸出5個類別的機率分佈

![CNN](images/cnn.png)  
*圖五：神經網路結構圖*  

### 訓練過程

```python
#模型訓練算法
def build_model():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 200, 200, 3], "x")
        y = tf.placeholder(tf.float32, [None, 5], "y")

    with tf.variable_scope("conv_layer_1"):
        conv1 = tf.layers.conv2d(x, 64, [3, 3], activation=tf.nn.relu, name='conv1')
        max1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
        bn1 = tf.layers.batch_normalization(max1, name='bn1')
        output1 = tf.layers.dropout(bn1, name='droput')

    with tf.variable_scope("conv_layer_2"):
        conv2 = tf.layers.conv2d(output1, 64, [3, 3], activation=tf.nn.relu, name='conv2')
        max2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2], name='max2')
        bn2 = tf.layers.batch_normalization(max2)
        output2 = tf.layers.dropout(bn2, name='dropout')

    with tf.variable_scope("conv_layer_3"):
        conv3 = tf.layers.conv2d(output2, 64, [3, 3], activation=tf.nn.relu, name='conv3')
        max3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2], name='max3')
        bn3 = tf.layers.batch_normalization(max3, name='bn3')
        output3 = bn3

    with tf.variable_scope("conv_layer_4"):
        conv4 = tf.layers.conv2d(output3, 32, [3, 3], activation=tf.nn.relu, name='conv4')
        max4 = tf.layers.max_pooling2d(conv4, [2, 2], [2, 2], name='max4')
        bn4 = tf.layers.batch_normalization(max4, name='bn4')
        output = bn4
        flatten = tf.layers.flatten(output, 'flatten')

    with tf.variable_scope("fc_layer1"):
        fc1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
        fc_bn1 = tf.layers.batch_normalization(fc1, name='bn1')
        dropout1 = tf.layers.dropout(fc_bn1, 0.5)

    with tf.variable_scope("fc_layer2"):
        fc2 = tf.layers.dense(dropout1, 128, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(fc2)

    with tf.variable_scope("fc_layer3"):
        fc3 = tf.layers.dense(dropout2, 64)
        dropout3 = tf.layers.dropout(fc3)

    with tf.variable_scope("fc_layer4"):
        fc4 = tf.layers.dense(dropout3, 32)

    with tf.variable_scope("fc_layer5"):
        fc5 = tf.layers.dense(fc4, 5)

    softmax = tf.nn.softmax(fc5, name='softmax')
    predict = tf.argmax(softmax, axis=1)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc5, labels=y, name='loss'))
    tf.summary.scalar("loss", loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(y, axis=1)), tf.float32))
    tf.summary.scalar("acc", accuracy)
    merged = tf.summary.merge_all()
    return x, y, predict, loss, accuracy, merged, softmax
```

---

## 圖像識別分類

### 即時辨識功能

#### 1. 圖片辨識模式
```python
#利用模型即時辨識影像
    def predict_value(self, type='image', image_path=None):
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        saver.restore(sess, tf.train.latest_checkpoint("./h5_dell1/"))
        if type == 'image':
            assert image_path is not None
            image = cv2.imread(image_path)
            image = cv2.resize(image, (200, 200))
            image = np.asarray(image, np.float32) / 255.
            image = np.reshape(image, (1, image.shape[ 0 ], image.shape[ 1 ], image.shape[ 2 ]))
            [ predict, probab ] = sess.run([ self.predict, self.probab ], feed_dict={self.x: image})
           # predict = sess.run(self.predict, feed_dict={self.x: image})
           # print("what? 1：",np.max(probab))
           # print("what? 2：",predict[0])
            return predict[0]
            if (np.max(probab)<1):
                print("recognise fail")
                predict=4
            print(predict)

        elif type == 'video':
            capture = cv2.VideoCapture(0)
            while True:
                ret, frame = capture.read()
                resize = cv2.resize(frame, (200, 200))
                x_ = np.asarray(resize, np.float32) / 255.
                x_ = np.reshape(x_, [ 1, x_.shape[ 0 ], x_.shape[ 1 ], x_.shape[ 2 ] ])
                [ predict, probab ] = sess.run([ self.predict, self.probab ], feed_dict={self.x: x_})
                if predict == 0:
                    cv2.putText(frame, "0 probab: %.3f" % np.max(probab), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 255), 2, cv2.LINE_AA)
                elif predict == 1:
                    cv2.putText(frame, "1 probab: %.3f" % np.max(probab), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 255), 2, cv2.LINE_AA)
                elif predict == 2:
                    cv2.putText(frame, "2 probab: %.3f" % np.max(probab), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 0), 2, cv2.LINE_AA)
                elif predict == 3:
                    cv2.putText(frame, "3 probab: %.3f" % np.max(probab), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 0, 255), 2, cv2.LINE_AA)
                elif predict == 4:
                    cv2.putText(frame, "4 probab: %.3f" % np.max(probab), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 0, 255), 2, cv2.LINE_AA)
                if predict==3:
                    print("1111")

                print(predict)

                cv2.imshow("recognized", frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cv2.destroyAllWindows()
            capture.release()
```

![result1](images/result1.png)  
*圖八：蔬菜類圖像辨識結果*  

![result2](images/result2.png)  
*圖九：易拉罐類圖片辨識效果*  



---

## 結果展示

### 性能指標

| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| Accuracy | 95.6% | 92.3% |
| Loss | 0.124 | 0.189 |

![Confusion Matrix](images/confusion.png)  
*圖十一：混淆矩陣*  

![Bar Image](images/accuracy.png)  
*圖十二：各類別辨識準確率長條圖*  

---

## 專案結構

```
cnn-image-classification/
├── picture/               # 數據集目錄
│   ├── 0-0.jpg
│   ├── 0-1.jpg
│   └── ...
├── h5_dell/              # 訓練模型保存目錄
│   ├── mode.ckpt.data
│   ├── mode.ckpt.index
│   └── ...
├── log/                  # TensorBoard日誌
├── data_preprocess.py    # 數據預處理腳本
├── train.py              # 模型訓練腳本
├── predict.py            # 圖像識別腳本
├── requirements.txt      # 依賴套件
├── notebook.ipynb        # Jupyter Notebook
└── README.md             # 項目說明
```

---


## 總結

本專案以實際案例介紹了神經網路影像辨識演算法的建構及使用詳細步驟，介紹了卷積神經網路實現影像辨識分類的詳細過程，以及實現效果的展示。

**項目亮點：**
- ✅ 完整的數據預處理流程
- ✅ 清晰的CNN網路架構
- ✅ 詳細的訓練過程記錄
- ✅ 支援圖片和視訊即時識別
- ✅ 適合Google Colab運行













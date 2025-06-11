import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image
import cv2
import os

mnist_bldr = tfds.builder("mnist")
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
mnist_train_orig = datasets["train"]
mnist_test_orig = datasets["test"]

# Prepare training, validation, and test data
def dataset_to_numpy(dataset):
    images, labels = [], []
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return (np.array(images), np.array(labels))

mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item["image"], tf.float32) / 255.0, item["label"])
)
mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item["image"], tf.float32) / 255.0, item["label"])
)

x_train, y_train = dataset_to_numpy(mnist_train)
x_test, y_test = dataset_to_numpy(mnist_test)

x_valid, y_valid = x_train[50000:-1], y_train[50000:-1]
x_train, y_train = x_train[0:50000], y_train[0:50000]

# Data Argumentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30 # -30 ~ +30度の範囲で回転
)
train_generator = datagen.flow(x_train, y_train, batch_size=64)

# Model Construction
if not os.path.exists("cnn_model.keras"):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=64, kernel_size=(5, 5), strides=(1, 1),
        padding="same", data_format="channels_last",
        name="conv_1", activation="relu"
    ))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2), name="pool_1"
    ))
    model.add(keras.layers.Conv2D(
        filters=128, kernel_size=(5, 5), strides=(1, 1),
        padding="same", data_format="channels_last",
        name="conv_2", activation="relu"
    ))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    tf.random.set_seed(1)
    model.build(input_shape=(None, 28, 28, 1))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Model training
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1
    )

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        callbacks=[es]
    )
else:
    model = keras.models.load_model("cnn_model.keras")

evaluate = model.evaluate(x_test, y_test, batch_size=64)
print(f"Test Accuracy: {evaluate[1]:.2%}")

model.save("cnn_model.keras")

def preprocess_digit(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化（背景を黒、数字を白に）
    # _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # **輪郭を取得**
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((280, 280), dtype=np.uint8)  # 何も描かれていない場合は空白画像を返す

    # **最大の輪郭（=数字が書かれている範囲）を取得**
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # if h > w:
    # **上下10%の余白を確保**
    padding_h = int(h * 0.1)  # 上下10%（hの10%）
    new_h = h + 2 * padding_h  # 上下に余白を追加

    # **新しいサイズ（正方形を作成する）**
    new_size = max(w, new_h)  # 幅と高さの大きい方を基準にする
    padded_digit = np.zeros((new_size, new_size), dtype=np.uint8)

    # **余白の中央配置**
    x_offset = (new_size - w) // 2
    y_offset = (new_size - new_h) // 2 + padding_h
    padded_digit[y_offset:y_offset + h, x_offset:x_offset + w] = thresh[y:y+h, x:x+w]

    # **280x280 にリサイズ**
    resized_digit = cv2.resize(padded_digit, (280, 280), interpolation=cv2.INTER_AREA)
    _, final_image = cv2.threshold(resized_digit, 0, 255, cv2.THRESH_BINARY_INV)

    return final_image
    # else:
    #     return image

img = Image.open("images/IMG_8787.jpg")
img = img.convert("L")
img = np.array(img)
img = preprocess_digit(img)
img = Image.fromarray((255 - img).astype('uint8'))  # 白黒反転
img = img.resize((28, 28))
img = np.array(img) / 255.0 # 正規化

img = img.reshape(1, 28, 28, 1)
pred = model.predict(img)
pred_class = np.argmax(pred)
print(f"判定結果: {pred_class} | 正解ラベル: 2")

img = Image.open("images/IMG_8790.PNG")
img = img.convert("L")
img = np.array(img)
img = preprocess_digit(img)
img = Image.fromarray((255 - img).astype('uint8'))  # 白黒反転
img = img.resize((28, 28))
img = np.array(img) / 255.0 # 正規化

img = img.reshape(1, 28, 28, 1)
pred = model.predict(img)
pred_class = np.argmax(pred)
print(f"判定結果: {pred_class} | 正解ラベル: 9")

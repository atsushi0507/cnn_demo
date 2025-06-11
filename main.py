import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2

if "model" not in st.session_state:
    # st.session_state.model = tf.keras.models.load_model("cnn_model.keras")
    st.session_state.model = tf.keras.models.load_model("cnn_model2.keras")
if "worst_model" not in st.session_state:
    st.session_state.worst_model = tf.keras.models.load_model("cnn_worst_model.keras")
if "rotate_model" not in st.session_state:
    st.session_state.rotate_model = tf.keras.models.load_model("cnn_model8.keras")

model = st.session_state.model
worst_model = st.session_state.worst_model
da_model = st.session_state.rotate_model

model = st.session_state.model
worst_model = st.session_state.worst_model

st.title("手書き数字認識デモ")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

def preprocess_digit(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化（背景を黒、数字を白に）
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # **輪郭を取得**
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((280, 280), dtype=np.uint8)  # 何も描かれていない場合は空白画像を返す

    # **最大の輪郭（=数字が書かれている範囲）を取得**
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    if h > w:
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
    else:
        return image


if st.button("判定する"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0] # チャンネルの抽出
        img = preprocess_digit(img)
        img = Image.fromarray((255 - img).astype('uint8'))  # 白黒反転
        img = img.resize((28, 28))
        img = np.array(img) / 255.0 # 正規化
        img = img.reshape(1, 28, 28, 1) # モデルの入力形状に合わせる
        st.image(img)

        # 推論
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        worst_pred = worst_model.predict(img)
        worst_pred_digit = np.argmax(worst_pred)

        da_pred = da_model.predict(img)
        da_pred_digit = np.argmax(da_pred)

        left, center, right = st.columns(3)
        with left:
            st.header("Best model")
            st.write(f"💻 **Best model: {predicted_digit}**")
            st.bar_chart(prediction[0]) # 確率分布を表示
        with center:
            st.header("Worst model")
            st.write(f"✏️ **Worst model: {worst_pred_digit}**")
            st.bar_chart(worst_pred[0])
        with right:
            st.header("Rotate model")
            st.write(f"🌀 **Rotate model: {da_pred_digit}**")
            st.bar_chart(da_pred[0])
    else:
        st.warning("手書き入力してください")

url_intro = "https://atsushi-datascience.com/blog/cnn-image-classification-basics"
url_advanced = "https://atsushi-datascience.com/blog/cnn-image-classification-advanced"
st.subheader("CNN に関する紹介ブログ")
st.write(f"[入門編]({url_intro})")
st.write(f"[実践編]({url_advanced})")

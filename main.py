import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.keras")
worst_model = tf.keras.models.load_model("cnn_worst_model.keras")

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

if st.button("判定する"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0] # チャンネルの抽出
        img = Image.fromarray((255 - img).astype('uint8'))  # 白黒反転
        img = img.resize((28, 28))
        img = np.array(img) / 255.0 # 正規化
        img = img.reshape(1, 28, 28, 1) # モデルの入力形状に合わせる

        # 推論
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        worst_pred = worst_model.predict(img)
        worst_pred_digit = np.argmax(worst_pred)

        left, right = st.columns(2)
        with left:
            st.header("Best model")
            st.write(f"💻 **Best model: {predicted_digit}**")
            st.bar_chart(prediction[0]) # 確率分布を表示
        with right:
            st.header("Worst model")
            st.write(f"✏️ **Worst model: {worst_pred_digit}**")
            st.bar_chart(worst_pred[0])
    else:
        st.warning("手書き入力してください")

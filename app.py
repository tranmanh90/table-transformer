from datetime import datetime

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")


st.sidebar.header("TABLE RECOGNITION DEMO")


uploaded_file = st.file_uploader(label="UPLOAD FILE", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")


from core import TableRecognizer

m = TableRecognizer(
    checkpoint_path="/data/pubtables1m/PubTables1M-Structure-PASCAL-VOC/output/20220110130310/model_11.pth"
)


def main():
    global image
    if image is None:
        return

    placeholder = st.image(image, width=500)  # display image

    with st.spinner("🤖 AI is at Work! "):
        start_time = datetime.now()
        results = m.predict(image_path=image)
        output_image = results["debug_image"]
        consume_time = datetime.now() - start_time

        st.text(f"Consume time: {consume_time}")

        placeholder.empty()
        st.image([image, output_image], caption=["input", "output"], width=500)
    st.balloons()


if __name__ == "__main__":
    main()

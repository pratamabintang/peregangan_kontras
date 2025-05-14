import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

st.title("📷 Simulasi Peregangan Kontras Media Grayscale")

@st.cache_data
def grayscale(img):
    row, col, _ = img.shape
    gray_img = np.zeros((row, col), dtype=np.uint8)
    for y in range(row):
        for x in range(col):
            B, G, R = img[y, x]
            gray_img[y, x] = int(0.114 * B + 0.587 * G + 0.299 * R)
    return gray_img

def contrast(img, r1, s1, r2, s2):
    row, col = img.shape
    stretch_img = np.zeros((row, col), dtype=np.uint16)
    y1 = s1 / r1 if r1 != 0 else 0
    y2 = (s2 - s1) / (r2 - r1) if r2 != r1 else 0
    y3 = (255 - s2) / (255 - r2) if r2 != 255 else 0

    for y in range(row):
        for x in range(col):
            if img[y, x] < r1:
                stretch_img[y, x] = y1 * img[y, x]
            elif img[y, x] > r2:
                stretch_img[y, x] = y3 * (img[y, x] - r2) + s2
            else:
                stretch_img[y, x] = y2 * (img[y, x] - r1) + s1

    return np.clip(stretch_img, 0, 255).astype(np.uint8)

upload_img = st.file_uploader("Pilih Gambar", type=['jpg', 'png', 'jpeg'])

if upload_img:
    raw_img = Image.open(upload_img)
    raw_img = np.array(raw_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_img, caption="Gambar Asli", use_container_width=True)

    row, col, channel = raw_img.shape

    img = grayscale(raw_img)
    with col2:
        st.image(img, caption="Gambar Grayscale", use_container_width=True)

    st.write("### Atur Parameter Peregangan Kontras")
    r1 = st.slider("Nilai r1", 0, 255, 80)
    s1 = st.slider("Nilai s1", 0, 255, 50)
    r2 = st.slider("Nilai r2", 0, 255, 200)
    s2 = st.slider("Nilai s2", 0, 255, 230)

    with st.spinner("Sedang memproses gambar..."):
        stretch_img = contrast(img, r1, s1, r2, s2)
    col1, col2 = st.columns(2)
    with col1:
        st.image(stretch_img, caption="Hasil Peregangan Kontras", use_container_width=True)

    x_vals = np.arange(256)
    y_vals = np.piecewise(x_vals,
        [x_vals < r1, (x_vals >= r1) & (x_vals <= r2), x_vals > r2],
        [lambda x: s1 / r1 * x if r1 != 0 else 0,
         lambda x: (s2 - s1) / (r2 - r1) * (x - r1) + s1 if r2 != r1 else 0,
         lambda x: (255 - s2) / (255 - r2) * (x - r2) + s2 if r2 != 255 else 0]
    )
    with col2:
        fig, ax = plt.subplots()
        ax.plot(x_vals, x_vals, 'r--')
        ax.plot(x_vals, y_vals, 'b-')
        ax.set_title("Grafik Peregangan Kontras")
        st.pyplot(fig)

    result_img = Image.fromarray(stretch_img)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    base_name, ext = os.path.splitext(upload_img.name)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            label="⬇️ Download Hasil Gambar",
            data=byte_im,
            file_name=f"{base_name}_stretch.png",
            mime="image/png"
        )
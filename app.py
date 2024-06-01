import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np
import os

def main():
    st.set_page_config(page_title="St Image processing app:Interactive Image Editing", layout="wide")
    st.title("Streamlit Interactive Image Editing")
    st.subheader("Upload an Image")
    ufile = st.file_uploader("Drag and drop file here or", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

    if ufile is not None:
        image = Image.open(ufile)
        st.image(image, caption='Original Image', use_column_width=True)

        st.sidebar.header("Tools")
        gs = st.sidebar.checkbox("Grayscale")
        blur = st.sidebar.checkbox("Blur")
        edges = st.sidebar.checkbox("Edge Detection")
        contrast_factor = st.sidebar.slider("Contrast", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        brightness_factor = st.sidebar.slider("Brightness", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        rotate_angle = st.sidebar.slider("Rotation", min_value=-180, max_value=180, value=0, step=1)
        flip_horizontal = st.sidebar.checkbox("Flip Horizontal")
        flip_vertical = st.sidebar.checkbox("Flip Vertical")
        sharpen = st.sidebar.checkbox("Sharpen")
        emboss = st.sidebar.checkbox("Emboss")
        color_adjust = st.sidebar.selectbox("Color Adjustment", ["None", "Invert", "Posterize", "Solarize"])

        # Image processing operations
        prc_img = image.copy()
        if gs:
            prc_img = ImageOps.grayscale(prc_img)
        if blur:
            blur_radius = st.sidebar.slider("Blur Radius", min_value=1, max_value=20, value=3, step=1)
            prc_img = prc_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if edges:
            prc_img = np.array(prc_img.convert('RGB'))
            prc_img = cv2.Canny(prc_img, 100, 200)
            prc_img = Image.fromarray(prc_img)
        if contrast_factor != 1.0:
            prc_img = ImageEnhance.Contrast(prc_img).enhance(contrast_factor)
        if brightness_factor != 1.0:
            prc_img = ImageEnhance.Brightness(prc_img).enhance(brightness_factor)
        if rotate_angle != 0:
            prc_img = prc_img.rotate(rotate_angle, expand=True)
        if flip_horizontal:
            prc_img = ImageOps.mirror(prc_img)
        if flip_vertical:
            prc_img = ImageOps.flip(prc_img)
        if sharpen:
            prc_img = prc_img.filter(ImageFilter.SHARPEN)
        if emboss:
            prc_img = prc_img.filter(ImageFilter.EMBOSS)
        if color_adjust == "Invert":
            prc_img = ImageOps.invert(prc_img)
        elif color_adjust == "Posterize":
            prc_img = ImageOps.posterize(prc_img, 4)
        elif color_adjust == "Solarize":
            prc_img = ImageOps.solarize(prc_img, threshold=128)

        st.subheader("Processed Image")
        st.image(prc_img, use_column_width=True)

        if st.button("Download Processed Image"):
            st.download_button("Download Processed Image", prc_img, file_name=f"processed_image_{ufile.name}")
    else:
        st.warning("Please upload an image here.")

    st.sidebar.subheader("Instructions")
    st.sidebar.info(
        "1. Upload an image by drag or drop or click the 'Browse files' button.\n"
        "2. Use the tools in the sidebar for image editing.\n"
        "3. The processed image will be displayed in real-time.\n"
        "4. Click 'Download Processed Image' to save the edited image.\n"
    )

if __name__ == "__main__":
    main()
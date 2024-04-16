import streamlit as st
import cv2
import numpy as np

# Function to resize the image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Function to convert the image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to crop the image
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

# Function to rotate the image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Load the image
def load_image(image_path):
    return cv2.imread(image_path)

# Create the Streamlit app
def main():
    st.title("Image Processing")

    # Load the image
    image_path = "dress1.jpg"
    image = load_image(image_path)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)

    # Image processing options
    st.sidebar.subheader("Image Processing Options")
    resize_option = st.sidebar.checkbox("Resize Image")
    grayscale_option = st.sidebar.checkbox("Convert to Grayscale")
    crop_option = st.sidebar.checkbox("Crop Image")
    rotation_option = st.sidebar.checkbox("Rotate Image")

    # Perform image processing based on selected options
    processed_image = image.copy()

    if resize_option:
        new_width = st.sidebar.slider("New Width", 50, 1000, 300)
        new_height = st.sidebar.slider("New Height", 50, 1000, 300)
        processed_image = resize_image(processed_image, new_width, new_height)
        st.subheader("Resized Image")
        st.image(processed_image, caption="Resized Image", use_column_width=True)

    if grayscale_option:
        processed_image = convert_to_grayscale(processed_image)
        st.subheader("Grayscale Image")
        st.image(processed_image, caption="Grayscale Image", use_column_width=True, channels='GRAY')

    if crop_option:
        x = st.sidebar.slider("X", 0, processed_image.shape[1], 0)
        y = st.sidebar.slider("Y", 0, processed_image.shape[0], 0)
        width = st.sidebar.slider("Width", 1, processed_image.shape[1], processed_image.shape[1])
        height = st.sidebar.slider("Height", 1, processed_image.shape[0], processed_image.shape[0])
        processed_image = crop_image(processed_image, x, y, width, height)
        st.subheader("Cropped Image")
        st.image(processed_image, caption="Cropped Image", use_column_width=True)

    if rotation_option:
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        processed_image = rotate_image(processed_image, angle)
        st.subheader("Rotated Image")
        st.image(processed_image, caption="Rotated Image", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()

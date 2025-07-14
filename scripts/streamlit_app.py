import os
import sys
import streamlit as st
from PIL import Image
import numpy as np

# Ensure project root is on sys.path so src package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.query import get_file_list

st.title("📷 CIFAR-10 Metadata Explorer")
st.sidebar.header("Filter Settings")

# Sidebar filters
min_bright    = st.sidebar.slider("Min Brightness", 0, 255, 100)
max_entropy   = st.sidebar.slider("Max Entropy",   0.0, 8.0, 4.5, step=0.1)
min_contrast  = st.sidebar.slider("Min Contrast",  0.0, 100.0, 0.0, step=0.1)
min_sharpness = st.sidebar.slider("Min Sharpness", 0.0, 500.0, 0.0, step=1.0)
limit         = st.sidebar.number_input("Number of Images", 1, 50, 9)

# Fetch filtered paths
paths = get_file_list(
    min_bright=min_bright,
    max_entropy=max_entropy,
    min_contrast=min_contrast,
    min_sharpness=min_sharpness,
    limit=limit
)

# Display images in a grid
cols = st.columns(int(np.ceil(limit ** 0.5)))
for idx, img_path in enumerate(paths):
    col = cols[idx % len(cols)]
    with col:
        st.image(Image.open(img_path), use_container_width=True)
        st.caption(os.path.basename(img_path))

if not paths:
    st.write("No images match these criteria.")
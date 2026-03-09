import streamlit as st
from gradio_client import Client
from PIL import Image
import requests
from io import BytesIO
import re

# Initialize Gradio Client
client = Client("ByteDance/Hyper-FLUX-8Steps-LoRA")

# Function to generate image
def generate_image(height, width, steps, scales, prompt, seed):
    try:
        # Call the model with provided parameters
        result = client.predict(
            height=height,
            width=width,
            steps=steps,
            scales=scales,
            prompt=prompt,
            seed=seed,
            api_name="/process_image"
        )
        return result
    except Exception as e:
        # Check for GPU quota error
        if "GPU quota" in str(e):
            # Extract the cooldown time from the error message
            match = re.search(r"Please retry in (\d+:\d+:\d+)", str(e))
            cooldown_time = match.group(1) if match else "unknown time"
            st.warning(
                f"The GPU is under cooldown and will be back within {cooldown_time}. "
                "Please visit https://hyperdyn.cloud for more information."
            )
        else:
            st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.set_page_config(page_title="Hyperdyn - Image Generation Tool", layout="centered")

st.title("Hyperdyn - Image Generation Tool")

# Define available resolutions with icons
resolutions = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "2048x2048": (2048, 2048)
}

# Input fields
selected_resolution = st.selectbox("Select resolution:", list(resolutions.keys()))
height, width = resolutions[selected_resolution]

# Basic settings
st.header("Basic Settings")
prompt = st.text_input("Enter your prompt:")

# Advanced settings
with st.expander("Advanced Settings"):
    steps = st.slider("Inference Steps", min_value=1, max_value=50, value=8)
    scales = st.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
    seed = st.number_input("Seed (for reproducibility)", min_value=0, max_value=10000, value=3413)

# Generate button
if st.button("Generate"):
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            # Generate the image
            image_path = generate_image(height, width, steps, scales, prompt, seed)
            
            if image_path:
                # Display the image preview
                try:
                    image = Image.open(image_path)
                    st.image(image, caption="Generated Image", use_column_width=True)
                    
                    # Provide a download button
                    with open(image_path, "rb") as file:
                        st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

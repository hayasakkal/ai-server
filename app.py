from flask import Flask, request, send_file
from PIL import Image
import io
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import os
import requests # Import the requests library

# --- AI Model Placeholder (for /edit endpoint) ---
pipe = None
model_loaded = False
print("--> Server starting WITHOUT loading the AI model initially.")
# -------------------------

# --- DeepAI Configuration (for /colorize endpoint) ---
DEEPAI_API_KEY = "293b8008-5ea2-4566-9ffb-be1fe6fdaaa2"
DEEPAI_COLORIZER_URL = "https://api.deepai.org/api/colorizer"
# ----------------------------------------------------


# This is the main Flask application
app = Flask(__name__)

def load_model():
    """Function to load the model when needed for the /edit endpoint."""
    global pipe, model_loaded
    if not model_loaded:
        print("--> First /edit request received. Loading InstructPix2Pix model now...")
        try:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float32,
                use_safetensors=True
            ).to("cpu")
            model_loaded = True
            print("--> AI Model for /edit loaded successfully!")
        except Exception as e:
            print(f"!!! FATAL: Failed to load AI model for /edit: {e} !!!")
            model_loaded = True 
            pipe = None
    return pipe


@app.route('/')
def home():
    """A simple endpoint to check if the server is running."""
    return "AI Server is running. It has two endpoints: /edit and /colorize."

@app.route('/colorize', methods=['POST'])
def colorize():
    """
    This new endpoint uses the DeepAI API to colorize an image.
    """
    if 'image' not in request.files:
        return "No image file found in the request.", 400

    print("--> /colorize endpoint called. Forwarding to DeepAI...")
    try:
        image_file = request.files['image']
        
        response = requests.post(
            DEEPAI_COLORIZER_URL,
            files={'image': image_file.read()},
            headers={'api-key': DEEPAI_API_KEY},
            timeout=120 # 2 minute timeout
        )

        print(f"--> DeepAI responded with status code: {response.status_code}")

        if response.status_code == 200:
            output_url = response.json().get('output_url')
            if output_url:
                # Download the colorized image from the URL provided by DeepAI
                image_response = requests.get(output_url, timeout=60)
                if image_response.status_code == 200:
                    print("--> Successfully downloaded colorized image from DeepAI.")
                    return image_response.content, 200, {'Content-Type': 'image/jpeg'}
                else:
                    return "Failed to download the processed image from DeepAI.", 500
            else:
                return "DeepAI did not return an output URL.", 500
        else:
            # Forward the error from DeepAI to the client
            return response.text, response.status_code

    except Exception as e:
        print(f"An error occurred while contacting DeepAI: {e}")
        return str(e), 500


@app.route('/edit', methods=['POST'])
def edit():
    """
    This endpoint uses a local model to edit an image.
    """
    active_pipe = load_model()
    if active_pipe is None:
        return "The AI model for editing is not available.", 503
    # ... (rest of the edit function remains the same)
    if 'image' not in request.files:
        return "No image file found in the request.", 400
    prompt = request.form.get('prompt')
    if not prompt:
        return "No prompt provided.", 400
    try:
        image_file = request.files['image']
        init_image = Image.open(image_file.stream).convert("RGB")
        print(f"--> Processing image with prompt: '{prompt}'")
        edited_image = active_pipe(prompt=prompt, image=init_image, num_inference_steps=15, image_guidance_scale=1.0).images[0]
        print("--> Image processed successfully!")
        buffer = io.BytesIO()
        edited_image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    except Exception as e:
        print(f"An error occurred during AI processing: {e}")
        return str(e), 500


# This is important for Render to know how to run the app.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

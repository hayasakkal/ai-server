from flask import Flask, request, send_file
from PIL import Image
import io
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import os

# --- AI Model Loading ---
# This will run when the server starts up on Render.
# It might be slow and consume a lot of memory.
print("--> Loading InstructPix2Pix model...")
pipe = None
try:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to("cpu")
    print("--> AI Model loaded successfully!")
except Exception as e:
    print(f"!!! FATAL: Failed to load AI model: {e} !!!")
# -------------------------


# This is the main Flask application
app = Flask(__name__)

@app.route('/')
def home():
    """A simple endpoint to check if the server is running."""
    if pipe is None:
        return "AI Server is running, but the AI MODEL FAILED TO LOAD. Check the logs.", 500
    return "AI Server is running and AI model is loaded!"

@app.route('/edit', methods=['POST'])
def edit():
    """
    This endpoint now uses the AI model to edit the image.
    """
    if pipe is None:
        return "The AI model is not available.", 503

    if 'image' not in request.files:
        return "No image file found in the request.", 400
    
    prompt = request.form.get('prompt')
    if not prompt:
        return "No prompt provided.", 400

    try:
        image_file = request.files['image']
        init_image = Image.open(image_file.stream).convert("RGB")

        print(f"--> Processing image with prompt: '{prompt}'")

        # Run the InstructPix2Pix model
        edited_image = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=15,
            image_guidance_scale=1.0,
        ).images[0]
        
        print("--> Image processed successfully!")

        # Save the edited image to a byte buffer
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

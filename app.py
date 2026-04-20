from flask import Flask, request, send_file
from PIL import Image
import io
import requests # Import the requests library

# --- DeepAI Configuration ---
DEEPAI_API_KEY = "293b8008-5ea2-4566-9ffb-be1fe6fdaaa2"
DEEPAI_COLORIZER_URL = "https://api.deepai.org/api/colorizer"
# ---------------------------


# This is the main Flask application
app = Flask(__name__)


@app.route('/')
def home():
    """A simple endpoint to check if the server is running."""
    return "AI Server for Colorization is running!"


@app.route('/colorize', methods=['POST'])
def colorize():
    """
    This endpoint uses the DeepAI API to colorize an image.
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
                image_response = requests.get(output_url, timeout=60)
                if image_response.status_code == 200:
                    print("--> Successfully downloaded colorized image from DeepAI.")
                    return image_response.content, 200, {'Content-Type': 'image/jpeg'}
                else:
                    return "Failed to download the processed image from DeepAI.", 500
            else:
                return "DeepAI did not return an output URL.", 500
        else:
            return response.text, response.status_code

    except Exception as e:
        print(f"An error occurred while contacting DeepAI: {e}")
        return str(e), 500


# This is important for Render to know how to run the app.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask, request, send_file
from PIL import Image
import io

# This is the main Flask application
app = Flask(__name__)

@app.route('/')
def home():
    """A simple endpoint to check if the server is running."""
    return "AI Server is running successfully on Render!"

@app.route('/edit', methods=['POST'])
def edit():
    """
    A test endpoint for image editing.
    For now, it just returns the same image it receives.
    """
    if 'image' not in request.files:
        return "No image file found in the request.", 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")

        # --- AI LOGIC WILL GO HERE LATER ---
        # For now, just return the original image to test the connection.
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        print("--> /edit endpoint was called, returning image.")
        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e), 500

# This is important for Render to know how to run the app.
if __name__ == "__main__":
    # Render provides the PORT environment variable.
    # The host must be '0.0.0.0' to be accessible externally.
    app.run(host="0.0.0.0", port=10000)

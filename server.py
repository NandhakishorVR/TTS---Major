from flask import Flask, request, send_file
import os

print("Starting server...")  # Debug print to ensure script is running

from inference import text_to_spectrogram  # If this fails silently, we'll know

app = Flask(__name__)

@app.route('/')
def home():
    return "Server is running!"

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return {"error": "No text provided"}, 400

    print(f"Received text: {text}")

    # Call your Tacotron synthesis pipeline
    text_to_spectrogram(text, output_name="out")  # will generate out.wav

    return send_file("out.wav", mimetype="audio/wav", as_attachment=True)

if __name__ == '__main__':
    print("Launching server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

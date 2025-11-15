from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
model = keras.models.load_model('mnist_model.keras')
print("Loaded Keras Model")


@app.route('/')
def home():
    return jsonify({"status": "api is running", "endpoint": "/predict"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image'] # Dictionary wartet um bilder von users zu empfangen
        img = Image.open(io.BytesIO(file.read())).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 784)

        prediction = model.predict(img_array, verbose=1)
        digits = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return jsonify({
            "digit": digits,
            "confidence": confidence
        }) # Python Dictionary -> JSON Object {String}

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Use environment PORT if available (for deployment), else 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port=port, debug=False)

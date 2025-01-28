from flask import Flask, request, jsonify, render_template
import onnxruntime
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms

app = Flask(__name__)

session = onnxruntime.InferenceSession('model.onnx')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4919, 0.4827, 0.4472], 
                       std=[0.2023, 0.1994, 0.2010])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        image_tensor = transform(image).unsqueeze(0).numpy()

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        outputs = session.run([output_name], {input_name: image_tensor})[0]
        
        probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]
        prediction_idx = np.argmax(probabilities[0])
        confidence = float(probabilities[0][prediction_idx])

        return jsonify({
            'predicted_class': classes[prediction_idx],
            'confidence': confidence,
            'class_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(classes, probabilities[0].tolist())
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)

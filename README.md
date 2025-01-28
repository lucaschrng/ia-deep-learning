# CIFAR-10 Image Classification Project

This project consists of two parts:
1. A Jupyter notebook for training the CIFAR-10 classification model
2. A web interface for using the trained model

## Setting Up the Development Environment

### Prerequisites
- Python 3.12
- pip

### Creating and Activating the Virtual Environment

1. Open a terminal in the project root directory and create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

### Notebook Sections
- Data loading and preprocessing
- Model architecture definition
- Training loop
- Model evaluation
- ONNX model export

## Web Interface

After training the model, you can use the web interface to test it:

1. Open `index.html` in a web browser
2. Upload an image or drag and drop one
3. Click "Predict" to see the classification results

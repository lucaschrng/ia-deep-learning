// Constants
const CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'];

const MODEL_PATH = 'model.onnx';

// Initialize elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');
const result = document.getElementById('result');
const predictedClass = document.getElementById('predicted-class');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const probabilities = document.getElementById('probabilities');

// Initialize ONNX session
let session;
async function initModel() {
    try {
        session = await ort.InferenceSession.create(MODEL_PATH);
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
    }
}
initModel();

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop zone when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropZone.classList.add('dragover');
}

function unhighlight() {
    dropZone.classList.remove('dragover');
}

// Handle dropped files
dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
        fileInput.files = dt.files;
    }
}

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        preview.classList.remove('hidden');
        predictBtn.disabled = false;
        result.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

async function preprocessImage(imageData) {
    // Create a canvas to resize the image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 32;
    canvas.height = 32;
    
    // Draw and resize the image
    ctx.drawImage(imageData, 0, 0, 32, 32);
    
    // Get image data and normalize
    const imageArray = ctx.getImageData(0, 0, 32, 32).data;
    const float32Data = new Float32Array(3 * 32 * 32);
    
    // Normalize using CIFAR-10 mean and std values
    const mean = [0.4919, 0.4827, 0.4472];
    const std = [0.2023, 0.1994, 0.2010];
    
    for (let i = 0; i < imageArray.length / 4; i++) {
        for (let c = 0; c < 3; c++) {
            const value = imageArray[i * 4 + c] / 255.0;
            float32Data[c * 32 * 32 + i] = (value - mean[c]) / std[c];
        }
    }
    
    return float32Data;
}

predictBtn.addEventListener('click', async () => {
    if (!session) {
        alert('Model is not loaded yet. Please try again in a moment.');
        return;
    }

    const file = fileInput.files[0];
    if (!file) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Processing...';

    try {
        // Load and preprocess the image
        const img = new Image();
        img.src = URL.createObjectURL(file);
        await new Promise((resolve) => img.onload = resolve);
        
        const preprocessedData = await preprocessImage(img);
        
        // Prepare the input tensor
        const tensor = new ort.Tensor('float32', preprocessedData, [1, 3, 32, 32]);
        
        // Run inference
        const outputs = await session.run({ 'input': tensor });
        const outputData = outputs[Object.keys(outputs)[0]].data;
        
        // Calculate softmax probabilities
        const expValues = outputData.map(x => Math.exp(x));
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        const probArray = expValues.map(x => x / sumExp);
        
        // Get prediction
        const predictionIdx = probArray.indexOf(Math.max(...probArray));
        const confidence = probArray[predictionIdx];

        // Update UI
        predictedClass.textContent = CLASSES[predictionIdx];
        const confidencePercentage = Math.round(confidence * 100);
        confidenceBar.style.width = `${confidencePercentage}%`;
        confidenceValue.textContent = `${confidencePercentage}%`;

        // Clear and update probabilities
        probabilities.innerHTML = '';
        CLASSES.map((className, idx) => ({
            class: className,
            prob: probArray[idx]
        }))
        .sort((a, b) => b.prob - a.prob)
        .forEach(({ class: className, prob }) => {
            const percentage = Math.round(prob * 100);
            const div = document.createElement('div');
            div.className = 'flex items-center space-x-2';
            div.innerHTML = `
                <div class="w-24 text-sm">${className}</div>
                <div class="flex-1 bg-gray-200 rounded-full h-2">
                    <div class="probability-bar bg-blue-600 h-2 rounded-full" style="width: ${percentage}%"></div>
                </div>
                <div class="w-12 text-sm text-right">${percentage}%</div>
            `;
            probabilities.appendChild(div);
        });

        result.classList.remove('hidden');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
});

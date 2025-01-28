document.addEventListener('DOMContentLoaded', function() {
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

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
            fileInput.files = dt.files; // Update the hidden file input
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

    predictBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        predictBtn.disabled = true;
        predictBtn.textContent = 'Processing...';

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // Update results
            predictedClass.textContent = data.predicted_class;
            const confidence = Math.round(data.confidence * 100);
            confidenceBar.style.width = `${confidence}%`;
            confidenceValue.textContent = `${confidence}%`;

            // Clear previous probabilities
            probabilities.innerHTML = '';

            // Add probability bars
            Object.entries(data.class_probabilities)
                .sort((a, b) => b[1] - a[1])
                .forEach(([className, prob]) => {
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
});

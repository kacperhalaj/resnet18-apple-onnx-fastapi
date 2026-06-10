from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import uvicorn

app = FastAPI(title="Apple Disease Classifier API")

ONNX_PATH = "apple_model.onnx"
ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

CLASS_NAMES = {
    0: "Apple Scab (Parch jabłoni)",
    1: "Apple Black Rot (Czarna zgnilizna)",
    2: "Apple Cedar Rust (Rdza jabłoniowo-cedrowa)",
    3: "Apple Healthy (Zdrowe)"
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# ==========================================
# Interfejs Graficzny (Strona Główna)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rozpoznawanie Chorób Jabłoni</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #10b981;
                --primary-hover: #059669;
                --bg: #f3f4f6;
                --text: #1f2937;
                --border: #d1d5db;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background: linear-gradient(135deg, #dcfce7, #f3f4f6); 
                color: var(--text); 
                min-height: 100vh; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                margin: 0; 
                padding: 20px;
                box-sizing: border-box;
            }
            .container { 
                background: white; 
                width: 100%; 
                max-width: 450px; 
                padding: 30px; 
                border-radius: 16px; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.08); 
                text-align: center; 
            }
            h1 { font-size: 1.5rem; margin-top: 0; margin-bottom: 10px; color: #111827; }
            p { color: #6b7280; font-size: 0.95rem; margin-bottom: 25px; line-height: 1.5; }
            .drop-zone { 
                border: 2px dashed var(--border); 
                border-radius: 12px; 
                padding: 40px 20px; 
                cursor: pointer; 
                transition: all 0.3s ease; 
                background-color: #f9fafb;
                position: relative;
            }
            .drop-zone.dragover { 
                border-color: var(--primary); 
                background-color: #ecfdf5; 
            }
            .drop-zone p { margin: 0; color: #4b5563; font-weight: 600; }
            .drop-zone span { font-size: 0.85rem; color: #9ca3af; font-weight: 400; }
            .drop-zone input[type="file"] { display: none; }
            .preview-container { 
                display: none; 
                margin-top: 20px; 
            }
            .preview-container img { 
                max-width: 100%; 
                max-height: 250px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                object-fit: cover;
            }
            .btn-group { 
                display: flex; 
                gap: 10px; 
                margin-top: 20px; 
            }
            button { 
                flex: 1; 
                padding: 12px; 
                border: none; 
                border-radius: 8px; 
                font-size: 1rem; 
                font-weight: 600; 
                cursor: pointer; 
                transition: 0.2s; 
            }
            .btn-primary { background-color: var(--primary); color: white; }
            .btn-primary:hover { background-color: var(--primary-hover); }
            .btn-primary:disabled { background-color: #a7f3d0; cursor: not-allowed; }
            .btn-secondary { background-color: #e5e7eb; color: #374151; }
            .btn-secondary:hover { background-color: #d1d5db; }
            #result { 
                margin-top: 20px; 
                font-size: 1rem; 
                padding: 15px; 
                border-radius: 8px; 
                display: none; 
                animation: fadeIn 0.3s ease;
            }
            .success { background-color: #ecfdf5; color: #065f46; border: 1px solid #34d399; }
            .error { background-color: #fef2f2; color: #991b1b; border: 1px solid #f87171; }
            .loading { background-color: #eff6ff; color: #1e3a8a; border: 1px solid #93c5fd; }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍎 Analiza Jabłoni</h1>
            <p>Wgraj zdjęcie liścia, aby sprawdzić jego stan zdrowia za pomocą modelu AI.</p>

            <div class="drop-zone" id="dropZone">
                <p>Przeciągnij zdjęcie tutaj</p>
                <span>lub kliknij, aby wybrać z dysku</span>
                <input type="file" id="imageInput" accept="image/*">
            </div>

            <div class="preview-container" id="previewContainer">
                <img id="imagePreview" src="" alt="Podgląd zdjęcia">
            </div>

            <div class="btn-group">
                <button class="btn-secondary" onclick="clearForm()">Wyczyść</button>
                <button class="btn-primary" id="analyzeBtn" onclick="analyzeImage()" disabled>Zbadaj zdjęcie</button>
            </div>

            <div id="result"></div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('imageInput');
            const previewContainer = document.getElementById('previewContainer');
            const imagePreview = document.getElementById('imagePreview');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const resultDiv = document.getElementById('result');

            let currentFile = null;

            dropZone.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) handleFile(e.target.files[0]);
            });

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    showResult('Proszę wybrać plik graficzny (zdjęcie).', 'error');
                    return;
                }
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    previewContainer.style.display = 'block';
                    dropZone.style.display = 'none';
                    analyzeBtn.disabled = false;
                    resultDiv.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }

            function clearForm() {
                currentFile = null;
                fileInput.value = '';
                previewContainer.style.display = 'none';
                imagePreview.src = '';
                dropZone.style.display = 'block';
                analyzeBtn.disabled = true;
                resultDiv.style.display = 'none';
            }

            function showResult(html, className) {
                resultDiv.style.display = 'block';
                resultDiv.className = className;
                resultDiv.innerHTML = html;
            }

            async function analyzeImage() {
                if (!currentFile) return;
                showResult('⏳ Trwa analizowanie...', 'loading');
                analyzeBtn.disabled = true;

                const formData = new FormData();
                formData.append('file', currentFile);

                try {
                    const response = await fetch('/predict/', { method: 'POST', body: formData });
                    if (response.ok) {
                        const data = await response.json();
                        showResult(`
                            <div style="font-size: 1.1em; margin-bottom: 5px;">
                                <b>Wynik:</b> ${data.class_name}
                            </div>
                            <div style="font-weight: 400;">
                                Pewność sieci: <b>${data.confidence}</b>
                            </div>
                        `, 'success');
                    } else {
                        throw new Error('Błąd serwera');
                    }
                } catch (error) {
                    showResult('❌ Wystąpił błąd podczas analizy obrazu.', 'error');
                } finally {
                    analyzeBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

# ==========================================
# API (Logika Przetwarzania)
# ==========================================
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
    onnx_output = ort_session.run(None, ort_inputs)[0][0]

    probabilities = softmax(onnx_output)
    class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[class_idx])

    return {
        "filename": file.filename,
        "class_id": class_idx,
        "class_name": CLASS_NAMES.get(class_idx, "Nieznana"),
        "confidence": f"{confidence * 100:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
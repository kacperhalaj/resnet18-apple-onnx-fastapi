import torch
import torch.nn as nn
from torchvision import models, transforms
import onnx
import onnxruntime as ort
import numpy as np
import time
from PIL import Image
import os

# ==========================================
# USTAWIENIA
# ==========================================

IMAGE_PATH = 'Apple_Disease_Dataset/test/Apple___Apple_scab/0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165_270deg.jpg'
MODEL_PATH = 'resnet18_apple_disease.pth'
ONNX_PATH = 'apple_model.onnx'
NUM_RUNS = 100

print("--- KROK 3: Eksport do ONNX ---")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()
print("Pomyślnie załadowano model PyTorch na CPU.")

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print(f"Model pomyślnie wyeksportowany do pliku: {ONNX_PATH}\n")

if not os.path.exists(IMAGE_PATH):
    print(f"[UWAGA] Nie znaleziono pliku testowego: {IMAGE_PATH}. Pomiary zostaną wykonane na losowym tensorze.")
    input_numpy = np.random.rand(1, 3, 224, 224).astype(np.float32)
    input_tensor = torch.from_numpy(input_numpy)
else:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_numpy = input_tensor.numpy()

print("--- KROK 4: Weryfikacja predykcji ---")
with torch.no_grad():
    pytorch_output = model(input_tensor).numpy()

ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
onnx_output = ort_session.run(None, ort_inputs)[0]

print(f"Wynik PyTorch (surowe logity): {pytorch_output[0]}")
print(f"Wynik ONNX    (surowe logity): {onnx_output[0]}")

np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05)
print("-> SUKCES: Predykcje PyTorch i ONNX są ze sobą zgodne!\n")

print(f"--- KROK 5: Pomiar czasu inferencji na CPU (średnia z {NUM_RUNS} prób) ---")
for _ in range(10):
    with torch.no_grad(): _ = model(input_tensor)

start_time = time.time()
for _ in range(NUM_RUNS):
    with torch.no_grad(): _ = model(input_tensor)
pytorch_time = (time.time() - start_time) / NUM_RUNS

for _ in range(10): _ = ort_session.run(None, ort_inputs)

start_time = time.time()
for _ in range(NUM_RUNS): _ = ort_session.run(None, ort_inputs)
onnx_time = (time.time() - start_time) / NUM_RUNS

print(f"Średni czas PyTorch CPU:       {pytorch_time * 1000:.2f} ms / zdjęcie")
print(f"Średni czas ONNX Runtime CPU:  {onnx_time * 1000:.2f} ms / zdjęcie")
# 🍎 Apple Disease Classifier

System rozpoznawania chorób liści jabłoni oparty na Transfer Learningu (ResNet18) z API w FastAPI i konteneryzacją Docker.

## O projekcie

Model AI klasyfikuje zdjęcia liści jabłoni do jednej z 4 klas:

| ID | Klasa | Opis |
|----|-------|------|
| 0 | Apple Scab | Parch jabłoni |
| 1 | Apple Black Rot | Czarna zgnilizna |
| 2 | Apple Cedar Rust | Rdza jabłoniowo-cedrowa |
| 3 | Apple Healthy | Zdrowe liście |

Model został wytrenowany na zbiorze [Apple Disease Dataset](https://www.kaggle.com/datasets/showravdhar/apple-disease-dataset) (7 771 zdjęć treningowych) i osiągnął **100% dokładności** na zbiorze walidacyjnym.

## Architektura

- **Model:** ResNet18 (Transfer Learning, 2-fazowy trening — Feature Extraction + Fine-Tuning)
- **Format produkcyjny:** ONNX Runtime (CPU) — ~2× szybszy od PyTorch
- **Backend:** FastAPI (asynchroniczne API REST)
- **Konteneryzacja:** Docker

## Wymagania

- Docker Desktop

## Uruchomienie

**1. Upewnij się, że w folderze projektu są cztery pliki:**
```
main.py
apple_model.onnx
requirements.txt
Dockerfile
```

**2. Zbuduj obraz:**
```bash
docker build -t apple-classifier .
```

**3. Uruchom kontener:**
```bash
docker run -d -p 8000:8000 --name moja_apka apple-classifier
```

**4. Otwórz w przeglądarce:**
- Interfejs użytkownika: http://127.0.0.1:8000/
- Dokumentacja API (Swagger): http://127.0.0.1:8000/docs

## Użycie API

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -F "file=@leaf.jpg"
```

Przykładowa odpowiedź:
```json
{
  "filename": "leaf.jpg",
  "class_id": 3,
  "class_name": "Apple Healthy (Zdrowe)",
  "confidence": "99.87%"
}
```

## Przydatne komendy Docker

```bash
# Logi aplikacji
docker logs moja_apka

# Zatrzymanie
docker stop moja_apka

# Pełna aktualizacja (po zmianie plików)
docker stop moja_apka && docker rm moja_apka
docker build -t apple-classifier .
docker run -d -p 8000:8000 --name moja_apka apple-classifier
```

## Struktura projektu

```
├── main.py              # Aplikacja FastAPI + interfejs webowy
├── projekt.py           # Skrypt trenowania modelu
├── test_onnx.py         # Eksport do ONNX i benchmarking
├── apple_model.onnx     # Wytrenowany model (produkcyjny)
├── requirements.txt     # Zależności Python
└── Dockerfile           # Konfiguracja kontenera
```

## Technologie

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)

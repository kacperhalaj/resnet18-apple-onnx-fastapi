# 1. Używamy lekkiego, oficjalnego obrazu Pythona
FROM python:3.12-slim

# 2. Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# 3. Kopiujemy plik z wymaganiami i instalujemy zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kopiujemy kod aplikacji oraz wyuczony model ONNX
COPY main.py .
COPY apple_model.onnx .

# 5. Otwieramy port 8000
EXPOSE 8000

# 6. Komenda uruchamiająca serwer API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
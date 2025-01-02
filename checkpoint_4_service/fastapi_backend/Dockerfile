FROM python:3.10-slim

WORKDIR /fastapi_backend

# Copy the backend code and requirements file
COPY . .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "python scripts/load_data.py && uvicorn fastapi_app:app --host 0.0.0.0 --port 8000"]

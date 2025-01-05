FROM python:3.10-slim

WORKDIR /streamlit_frontend

# Copy the frontend code and requirements file
COPY . .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]

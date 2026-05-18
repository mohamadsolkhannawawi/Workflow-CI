FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY model/ /app/model/

# Copy serving script
COPY serve.py .

EXPOSE 8080

CMD ["python", "serve.py"]

# Use Python 3.10 (or your target version)
FROM python:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# Set working dir
WORKDIR /app

# Copy app code
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 6969

# Launch FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6969", "--reload"]
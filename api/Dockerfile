# Use Python 3.10 (or your target version)
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# Set working dir
WORKDIR /app

COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . /app

RUN pip install --no-cache-dir -e ./MobileSAM

EXPOSE 6969

# Launch FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6969"]
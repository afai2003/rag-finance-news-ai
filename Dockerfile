# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

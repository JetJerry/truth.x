# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (FFmpeg for video/audio, libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port (Render/Railway use generic ports, HF uses 7860)
EXPOSE 7860
EXPOSE 8000

# Command to run the application
# We use the python module syntax to run uvicorn
CMD ["uvicorn", "TRUTH_Xx.backend.app:app", "--host", "0.0.0.0", "--port", "7860"]

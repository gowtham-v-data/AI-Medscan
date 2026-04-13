# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Include a label to identify the image
LABEL maintainer="AI MedScan Platform"

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable oneDNN warnings and ensure TF uses CPU optimally
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PORT=7860

# Install essential system dependencies (if using regular OpenCV, else headless is fine)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY backend/requirements.txt /app/backend/

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application files to the container
# This includes both the /backend and /frontend directories
COPY . /app/

# Expose the API port
EXPOSE 7860

# Change working directory so uvicorn can resolve "app.main:app"
WORKDIR /app/backend

# Command to run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

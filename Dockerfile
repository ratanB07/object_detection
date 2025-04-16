# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install gunicorn
RUN pip install gunicorn

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]

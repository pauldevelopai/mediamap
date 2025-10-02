# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including WeasyPrint dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libcairo-gobject2 \
    libcairo2 \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/instance
RUN mkdir -p /app/backend/training/models
RUN mkdir -p /app/_aimap_media
RUN mkdir -p /app/_aimap_reports

# Set environment variables
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "backend.app:app"] 
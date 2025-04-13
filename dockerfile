# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install UV and dependencies
RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml
RUN uv pip install --system -e .

# Copy the rest of the application
COPY . .

# Install PyTorch with CPU support only (to keep image smaller)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Make port 5000 available for Flask app
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0

# Run the application
CMD ["python", "app.py"]
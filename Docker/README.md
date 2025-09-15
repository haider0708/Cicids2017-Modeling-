# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)
```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop the service
docker-compose down
```

### Using Docker directly
```bash
# Build the image
docker build -t rihem-network-detection .

# Run the container
docker run -p 8000:8000 --name rihem-api rihem-network-detection

# Run in background
docker run -d -p 8000:8000 --name rihem-api rihem-network-detection
```

## Access the API
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Features
- Uses Python 3.11 slim image for minimal size
- Non-root user for security
- Health check endpoint
- Optimized layer caching
- Automatic restart policy

## Notes
- The container expects trained models in the `Models/` directory
- Make sure all required model files exist before building
- The API runs on port 8000 inside the container

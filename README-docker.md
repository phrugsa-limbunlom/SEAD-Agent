# Docker Deployment Guide

This guide explains how to run the chatbot-app and chatbot-server as separate Docker containers.

## Prerequisites

- Docker and Docker Compose installed on your system
- Environment variables configured

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Mistral AI API Key (required for chatbot-server)
MISTRAL_API_KEY=your_mistral_api_key_here

# React App API URL (for development)
REACT_APP_API_URL=http://localhost:8000
```

## Running with Docker Compose

### 1. Build and Run Both Services

```bash
# Build and start both services
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build
```

### 2. Access the Application

- **Frontend (React App)**: http://localhost:3000
- **Backend (FastAPI Server)**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. Stop the Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Running Individual Services

### Frontend Only

```bash
# Build and run the React app
cd chatbot-app
docker build -t chatbot-app .
docker run -p 3000:3000 -e REACT_APP_API_URL=http://localhost:8000 chatbot-app
```

### Backend Only

```bash
# Build and run the FastAPI server
cd chatbot-server
docker build -t chatbot-server .
docker run -p 8000:8000 -e MISTRAL_API_KEY=your_api_key_here chatbot-server
```

## Development Mode

For development with hot reload:

```bash
# Start services with volume mounts for development
docker-compose up --build
```

The `docker-compose.yaml` file includes volume mounts that enable:
- Live code reloading for both frontend and backend
- Persistent node_modules for the React app
- Development-friendly environment

## Testing

To run tests in the containerized environment:

```bash
# Run backend tests
docker-compose exec chatbot-server python tests/run_tests.py

# Or run tests in a separate container
docker run --rm -v $(pwd)/tests:/app/tests chatbot-server python tests/run_tests.py
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 3000 and 8000 are not in use
2. **Environment variables**: Ensure the `.env` file is properly configured
3. **API connection**: The frontend connects to the backend using the REACT_APP_API_URL environment variable

### Logs

```bash
# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs chatbot-server
docker-compose logs chatbot-app

# Follow logs in real-time
docker-compose logs -f
```

## Network Configuration

The services communicate through a custom Docker network called `chatbot-network`. This allows:
- Secure communication between frontend and backend
- Isolated network environment
- Service discovery by container name

## Production Deployment

For production deployment, consider:

1. **Environment-specific configuration**
2. **SSL/TLS certificates**
3. **Reverse proxy (nginx)**
4. **Database persistence**
5. **Monitoring and logging**

Example production docker-compose override:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  chatbot-app:
    command: npm run build && serve -s build -l 3000
    environment:
      - NODE_ENV=production
      - REACT_APP_API_URL=https://your-api-domain.com

  chatbot-server:
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
    environment:
      - ENVIRONMENT=production
```

Run with: `docker-compose -f docker-compose.yaml -f docker-compose.prod.yml up`

## Benefits of Separate Services

This separate service approach provides significant advantages:

- **Better development workflow** with hot reload for both frontend and backend
- **Independent scaling** of frontend and backend services  
- **Clearer separation of concerns** between React app and FastAPI server
- **Easier debugging** with separate logs and processes
- **More flexible deployment options** in production environments
- **Better resource utilization** with separate container management
- **Simplified CI/CD** with independent service deployment
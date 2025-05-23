FROM node:23-alpine AS frontend-build
WORKDIR /app
COPY chatbot-app/ /app
RUN npm install && npm run build

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
WORKDIR /home/src

RUN apt-get update && apt-get install -y curl

COPY chatbot-server/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY chatbot-server/ backend/

COPY --from=frontend-build /app/build/ backend/frontend/build/

COPY tests/ tests/

ENV PYTHONPATH=/home/src/backend

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh


EXPOSE 8080


ENTRYPOINT ["./docker-entrypoint.sh"]
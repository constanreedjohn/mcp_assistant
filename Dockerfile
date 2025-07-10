FROM python:3.11-slim

ENV MCP_CLIENT_URL=http://localhost:5001
ENV OLLAMA_LLM_URL=http://localhost:11434
ENV IMAGE_GEN_URL=http://localhost:3001

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls -l /app

ENV PYTHONPATH=/app

EXPOSE 5001
EXPOSE 8080
EXPOSE 3001
EXPOSE 7860

# Default command can be overridden by docker-compose
# CMD ["python", "servers/app.py"]

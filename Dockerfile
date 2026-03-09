# Fineas.ai — Railway deployment image
# Runs the web dashboard (chat_app.py) — NOT the MCP server
# ~200MB image (python:3.11-slim + core deps, no torch/transformers/fastmcp)

FROM python:3.11-slim

# System deps for building Python C extensions (pymongo, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Verify the app can import (fail fast if deps are broken)
RUN python -c "from sec_mcp.chat_app import app; print('OK: chat_app imports successfully')"

# Railway sets PORT env var; default to 8877 for local Docker runs
ENV PYTHONUNBUFFERED=1
ENV PORT=8877

EXPOSE 8877

# Health check with startup grace period
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["python", "-m", "sec_mcp.chat_app"]

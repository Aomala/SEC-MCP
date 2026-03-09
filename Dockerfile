# SEC Terminal — Railway deployment image
# Uses Python 3.11 slim to keep image small (~200MB vs 2.5GB with torch)
# NLP models (torch/transformers) are NOT included — too heavy for Railway.
# All financial extraction, SEC EDGAR API, MongoDB, and Chart.js UI work without them.

FROM python:3.11-slim

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY .env.example .env.example

# Install the package (editable mode so module resolution works)
RUN pip install --no-cache-dir -e .

# Health check: verify the app can import
RUN python -c "from sec_mcp.chat_app import app; print('Import check passed')"

# Railway sets PORT env var; default to 8877 for local Docker runs
ENV PYTHONUNBUFFERED=1
ENV PORT=8877

EXPOSE 8877

# Health check endpoint is at /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run the chat app (reads PORT from env)
CMD ["python", "-m", "sec_mcp.chat_app"]

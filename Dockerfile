# Fineas.ai — Railway deployment image
# Runs the web dashboard (chat_app.py) — NOT the MCP server

FROM python:3.11-slim

# System deps for building Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies directly (skip hatchling build system entirely)
RUN pip install --no-cache-dir \
    requests>=2.31 \
    beautifulsoup4>=4.12 \
    "pandas>=2.0" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0" \
    "fastapi>=0.100" \
    "uvicorn>=0.20" \
    "pymongo[srv]>=4.6" \
    "anthropic>=0.40" \
    "yfinance>=0.2.30" \
    "python-dotenv>=1.0"

# Copy source code
COPY src/ src/

# Add source to Python path (no package install needed)
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8877

# Verify imports work
RUN python -c "from sec_mcp.chat_app import app; print('OK')"

EXPOSE 8877

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["python", "-m", "sec_mcp.chat_app"]

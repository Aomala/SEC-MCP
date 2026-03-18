# Fineas.ai — Railway deployment image
# Runs the web dashboard (chat_app.py) — NOT the MCP server

FROM python:3.11-slim

# System deps for building Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies directly (skip hatchling build system entirely)
RUN pip install --no-cache-dir \
    "requests>=2.31" \
    "beautifulsoup4>=4.12" \
    "pandas>=2.0" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0" \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.20" \
    "httpx>=0.25" \
    "anthropic>=0.40" \
    "yfinance>=0.2.30" \
    "python-dotenv>=1.0" \
    "supabase>=2.0"

# Copy source code
COPY src/ src/

# Add source to Python path (no package install needed)
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Verify imports work at build time
RUN python -c "from sec_mcp.chat_app import app; print('imports OK')"

# Railway sets PORT dynamically — app reads os.environ["PORT"] with fallback to 8877
CMD ["python", "-m", "sec_mcp.chat_app"]

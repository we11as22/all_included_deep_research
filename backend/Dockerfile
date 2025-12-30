FROM python:3.11-slim-bookworm

# Set locale to UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -o Acquire::Retries=3 && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy requirements and source code for package installation
COPY pyproject.toml ./
COPY src ./src

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -e .[huggingface] && \
    pip install --no-cache-dir playwright || true

# Install Playwright browsers (if playwright is installed)
RUN python -c "import playwright; playwright.install()" 2>/dev/null || echo "Playwright not installed, skipping browser installation"

# Copy application code (alembic)
COPY alembic.ini ./
COPY alembic ./alembic

# Create memory directory
RUN mkdir -p /memory_files

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run migrations and start server
CMD alembic upgrade head && python -m src

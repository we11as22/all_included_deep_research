FROM python:3.11-slim-bookworm

# Set locale to UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies including searxng build dependencies
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
    python3-dev \
    python3-babel \
    python3-venv \
    python-is-python3 \
    uwsgi \
    uwsgi-plugin-python3 \
    git \
    build-essential \
    libxslt-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy requirements and source code for package installation
COPY backend/pyproject.toml ./
COPY backend/src ./src

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -e .[huggingface,playwright]

# Install Playwright browsers
RUN playwright install chromium && \
    playwright install-deps chromium || true

# Copy application code (alembic)
COPY backend/alembic.ini ./
COPY backend/alembic ./alembic

# Create memory directory
RUN mkdir -p /memory_files

# Setup searxng user and directories (similar to Perplexica)
RUN useradd --shell /bin/bash --system \
    --home-dir "/usr/local/searxng" \
    --comment 'Privacy-respecting metasearch engine' \
    searxng

RUN mkdir "/usr/local/searxng"
RUN mkdir -p /etc/searxng
RUN chown -R "searxng:searxng" "/usr/local/searxng"

# Copy searxng configuration files
COPY docker/searxng/settings.yml /etc/searxng/settings.yml
COPY docker/searxng/limiter.toml /etc/searxng/limiter.toml
COPY docker/searxng/uwsgi.ini /etc/searxng/uwsgi.ini
RUN chown -R searxng:searxng /etc/searxng

# Switch to searxng user to clone and setup searxng
USER searxng

# Clone searxng repository
RUN git clone "https://github.com/searxng/searxng" \
                   "/usr/local/searxng/searxng-src"

# Create Python virtual environment for searxng
RUN python3 -m venv "/usr/local/searxng/searx-pyenv"
RUN "/usr/local/searxng/searx-pyenv/bin/pip" install --upgrade pip setuptools wheel pyyaml msgspec
RUN cd "/usr/local/searxng/searxng-src" && \
    "/usr/local/searxng/searx-pyenv/bin/pip" install --use-pep517 --no-build-isolation -e .

# Switch back to root
USER root

# Copy entrypoint script
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh || true

# Allow searxng user to use sudo (needed for flask run)
RUN echo "searxng ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Expose ports
EXPOSE 8000 8080

# Set environment variable for searxng
ENV SEARXNG_API_URL=http://localhost:8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run entrypoint script
CMD ["/entrypoint.sh"]

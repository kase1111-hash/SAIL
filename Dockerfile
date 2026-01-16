# SAIL Development Dockerfile
# Privacy-first: No telemetry, no cloud dependencies

FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for audio and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio dependencies
    portaudio19-dev \
    libsndfile1 \
    libasound2-dev \
    # Build dependencies
    build-essential \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base AS development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash sail
USER sail

# Set up Python environment
ENV PATH="/home/sail/.local/bin:$PATH"

# Copy dependency files
COPY --chown=sail:sail pyproject.toml ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -e ".[dev]"

# Copy source code
COPY --chown=sail:sail . .

# Install the package in editable mode
RUN pip install --user --no-cache-dir -e ".[dev]"

# Default command
CMD ["bash"]

# Production stage
FROM base AS production

# Create non-root user
RUN useradd --create-home --shell /bin/bash sail
USER sail

ENV PATH="/home/sail/.local/bin:$PATH"

# Copy only necessary files
COPY --chown=sail:sail pyproject.toml ./
COPY --chown=sail:sail src/ ./src/
COPY --chown=sail:sail data/ ./data/
COPY --chown=sail:sail models/ ./models/

# Install production dependencies only
RUN pip install --user --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sail; print('healthy')" || exit 1

# Run SAIL
CMD ["sail", "run"]

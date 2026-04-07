FROM python:3.11-slim

LABEL maintainer="sre-bench"
LABEL description="SRE-Bench: Production Incident Response OpenEnv"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY sre_bench/ ./sre_bench/
COPY baseline.py .
COPY openenv.yaml .
COPY app.py .

# Expose Gradio port
EXPOSE 7860

# Default: run the Gradio demo
# Override with: docker run sre-bench python baseline.py
CMD ["python", "app.py"]

FROM python:3.10-slim

# Set working directory
WORKDIR /k8s

# Copy code
COPY app.py .
COPY outputs ./outputs

# Install dependencies
RUN pip install fastapi uvicorn joblib numpy scikit-learn prometheus-client

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install dependencies (including scikit-learn)
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy lightgbm scikit-learn pydantic

# Run FastAPI server
CMD ["uvicorn", "wine_api:app", "--host", "0.0.0.0", "--port", "8000"]

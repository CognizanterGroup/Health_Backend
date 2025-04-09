FROM python:3.11-slim-bullseye

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a directory for the database
RUN mkdir -p /app/data
ENV DATABASE_URL=sqlite:///./data/runpod_jobs.db

# Use the PORT environment variable from Render
ENV PORT=8000

# Command to run the application (using $PORT which Render sets automatically)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
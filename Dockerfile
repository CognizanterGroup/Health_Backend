FROM python:3.11-slim-bullseye

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code m
COPY . .

# Create a directory for the database
RUN mkdir -p /app/data
ENV DATABASE_URL=sqlite:///./data/runpod_jobs.db
ENV WEBHOOK=https://health-services-43i9.onrender.com/webhook
ENV RUNPOD_API_KEY=rpa_UEUBTC8Q6SL58YSXXFPDW37XAPGCCGNL38BIPLUL1k3i9i
ENV RUNPOD_ENDPOINT_ID=fswz4ju3asche1
ENV PUBLIC_URL=https://health-services-43i9.onrender.com

# Use the PORT environment variable from Render
ENV PORT=8000

# Command to run the application (using $PORT which Render sets automatically)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Flask
ENV PORT=8080
ENV PYTHONUNBUFFERED=True

# Expose the port Flask will run on
EXPOSE 8080

# Start Flask app (replace with your actual app filename and Flask app name)
CMD ["python", "app-backend.py"]

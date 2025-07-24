FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Environment variable
ENV PORT=8080

# Expose the port Flask will run on
EXPOSE 8080

# Start Flask app (replace with your actual app filename and Flask app name)
CMD ["python", "app-backend.py"]

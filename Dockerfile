# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the PYTHONPATH to include the root directory
ENV PYTHONPATH=/app

# Expose the application port for HuggingFace
EXPOSE 7860

# Run the application (Production Mode)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

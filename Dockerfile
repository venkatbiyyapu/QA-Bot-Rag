# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and install packages in the venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Make sure the virtual environment is used
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "gemini_qa_bot.py", "--server.port=8501", "--server.address=0.0.0.0"]

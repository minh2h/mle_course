FROM python:3.12-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        procps \
        curl \
        wget \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*
    
# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the ports
EXPOSE 8888 8080 4040

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes
ENV PYTHONPATH=/app
ENV SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

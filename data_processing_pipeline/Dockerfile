FROM python:3.11-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 - available in Debian 11)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh
    
# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the default JupyterLab port
EXPOSE 8888

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

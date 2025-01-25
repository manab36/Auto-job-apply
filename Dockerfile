# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    python3 python3-pip \
    wget unzip xvfb \
    libglib2.0-0 libnss3 libx11-xcb1 libatk1.0-0 libgtk-3-0 libxcomposite1 \
    libxrandr2 libasound2 libxdamage1 libgbm1 \
    cron \
    && rm -rf /var/lib/apt/lists/*  # Clean up package lists to reduce image size

# Install Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt update && apt install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*
    

# Set environment variable for virtual display
ENV DISPLAY=:99

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install required Python libraries
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run Xvfb and the Python script
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & python3 main.py"]

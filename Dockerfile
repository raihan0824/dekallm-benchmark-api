FROM python:3.11-slim

# Create user 1000
RUN useradd -u 1000 -m -s /bin/bash appuser

# Set the working directory
WORKDIR /opt/app

# Copy the requirements file and install the requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . ./

# Expose port 8000
EXPOSE 8000

# Change ownership of the working directory to user 1000
RUN chown -R 1000:1000 /opt/app

# Switch to user 1000
USER 1000

# Set the command to run the application
CMD ["python3", "main.py"]
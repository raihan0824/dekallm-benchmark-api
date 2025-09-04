FROM python:3.11-slim

# Set the working directory
WORKDIR /opt/app

# Copy the requirements file and install the requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . ./

# Expose port 8000
EXPOSE 8000

# Set the command to run the application
CMD ["python3", "main.py"]
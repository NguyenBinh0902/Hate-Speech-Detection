# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the embeddings_index-001.pkl file
RUN apt-get update && apt-get install -y wget
RUN wget -O embeddings_index-001.pkl "https://firebasestorage.googleapis.com/v0/b/jewelry-shop-781c0.appspot.com/o/embeddings_index-001.pkl?alt=media&token=9365f5e7-71af-405d-96d1-5281850fdb51"

# Expose the port that the app runs on
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=my_api.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
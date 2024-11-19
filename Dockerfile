# SAMPLE DOCKER FILE
# yuvrajmakkena/hdt:gpu-ultralytics-yolov5
# Base image
# FROM ultralytics/yolov5:latest
FROM nvcr.io/nvidia/tensorrt:21.06-py3

# Set the working directory in the container
WORKDIR /app

RUN apt update && apt install python3-pip -y && apt install ffmpeg -y 

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-binary :all: flask torch torchvision opencv-python-headless numpy pycuda
# Copy the server code and yolo model into the container 
#COPY best.pt .
COPY traffic_signs_a40n.engine .
COPY yolov5_trt.py .
COPY server.py .
# Start the server
CMD ["python3", "server.py"]

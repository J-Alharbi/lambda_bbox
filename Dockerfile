FROM public.ecr.aws/lambda/python:3.10

# Install system packages required by OpenCV
RUN yum install -y gcc ffmpeg libsm6 libxext6 git

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python-headless matplotlib boto3

# Copy your Lambda handler
COPY app.py ${LAMBDA_TASK_ROOT}

# Command for Lambda
CMD ["app.handler"]

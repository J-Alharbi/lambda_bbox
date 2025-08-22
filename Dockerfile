FROM public.ecr.aws/lambda/python:3.12

# Install system packages required by OpenCV
RUN yum install -y gcc ffmpeg libsm6 libxext6 git

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python-headless matplotlib boto3

# Copy your Lambda handler
COPY app.py ${LAMBDA_TASK_ROOT}

RUN mkdir -p ${LAMBDA_TASK_ROOT}/models && \
    aws s3 cp s3://model-bucket-22/product_best.pt ${LAMBDA_TASK_ROOT}/product_best.pt && \
    aws s3 cp s3://model-bucket-22/pricetag_best.pt ${LAMBDA_TASK_ROOT}/pricetag_best.pt

# Command for Lambda
CMD ["app.handler"]

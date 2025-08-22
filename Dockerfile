FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies required for OpenCV
RUN dnf install -y \
    gcc gcc-c++ git \
    libSM libXext libX11-devel libXrandr libXinerama libXcursor \
    mesa-libGL mesa-libGLU libpng zlib \
    && dnf clean all

# Install Python dependencies into Lambda task root
RUN pip install --upgrade pip
RUN pip install --no-cache-dir boto3 ultralytics opencv-python-headless numpy --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler
COPY app.py ${LAMBDA_TASK_ROOT}

# Create models directory (Lambda handler will download models at runtime)
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models

# Set Lambda handler
CMD ["app.handler"]

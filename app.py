import boto3
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import re

# S3 client
s3 = boto3.client("s3")

# Buckets
OUTPUT_BUCKET = "bbox-frames-1"
MODEL_BUCKET = "model-bucket-22"

# Models paths inside Lambda
MODELS_DIR = "/var/task/models"
os.makedirs(MODELS_DIR, exist_ok=True)

PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, "product_best.pt")
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, "pricetag_best.pt")

# Download models from S3 if not already present
def download_model_from_s3(model_name, local_path):
    if not os.path.exists(local_path):
        s3.download_file(MODEL_BUCKET, model_name, local_path)

download_model_from_s3("product_best.pt", PRODUCT_MODEL_PATH)
download_model_from_s3("pricetag_best.pt", PRICE_MODEL_PATH)

# Load YOLO models at cold start
model_product = YOLO(PRODUCT_MODEL_PATH)
model_price = YOLO(PRICE_MODEL_PATH)


def draw_boxes(img, results, label_name):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        color = (0, 255, 0) if label_name == "Product" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label_name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return img


def extract_video_name(filename: str) -> str:
    """
    Extract video name from filename.
    Example: chips-t4K_t00006500_var887_hq.jpg -> chips-t4K
    """
    match = re.match(r"([^_]+)", filename)
    return match.group(1) if match else "unknown"


def handler(event, context):
    # Extract bucket and key from event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    filename = os.path.basename(key)
    video_name = extract_video_name(filename)

    # Download input image to /tmp
    tmp_in_path = os.path.join(tempfile.gettempdir(), filename)
    s3.download_file(bucket, key, tmp_in_path)

    # Load image
    frame = cv2.imread(tmp_in_path)

    # Run inference
    product_results = model_product(frame)[0]
    price_results = model_price(frame)[0]

    # Draw detections
    img = draw_boxes(frame.copy(), product_results, "Product")
    img = draw_boxes(img, price_results, "PriceTag")

    # Save annotated image to /tmp
    tmp_out_path = os.path.join(tempfile.gettempdir(), f"annotated_{filename}")
    cv2.imwrite(tmp_out_path, img)

    # Upload annotated image to S3
    output_key = f"{video_name}/{filename}"
    s3.upload_file(tmp_out_path, OUTPUT_BUCKET, output_key)

    return {
        "statusCode": 200,
        "body": f"Annotated image saved to s3://{OUTPUT_BUCKET}/{output_key}"
    }

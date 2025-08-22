import boto3
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import re

s3 = boto3.client("s3")
OUTPUT_BUCKET = "bbox-frames-1"


# Load YOLO models once at cold start
model_product = YOLO("/var/task/models/product_best.pt")
model_price = YOLO("/var/task/models/pricetag_best.pt")


def draw_boxes(img, results, label_name):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        color = (0, 255, 0) if label_name == "Product" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img,
                    f"{label_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)
    return img

def extract_video_name(filename: str) -> str:
    """
    Extract video name from filename.
    Example: chips-t4K_t00006500_var887_hq.jpg -> chips-t4K
    """
    match = re.match(r"([^_]+)", filename)  # everything before first "_"
    return match.group(1) if match else "unknown"

def handler(event, context):
    # Extract bucket and key from event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    filename = os.path.basename(key)
    video_name = extract_video_name(filename)

    # Download image
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    s3.download_file(bucket, key, tmp_in.name)

    # Load image
    frame = cv2.imread(tmp_in.name)

    # Run inference
    product_results = model_product(frame)[0]
    price_results = model_price(frame)[0]

    # Draw detections
    img = draw_boxes(frame.copy(), product_results, "Product")
    img = draw_boxes(img, price_results, "PriceTag")

    # Save output
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp_out.name, img)

    # Upload annotated image to folder named after video
    output_key = f"{video_name}/{filename}"
    s3.upload_file(tmp_out.name, OUTPUT_BUCKET, output_key)

    return {
        "statusCode": 200,
        "body": f"Annotated image saved to s3://{bucket}/{output_key}"
    }

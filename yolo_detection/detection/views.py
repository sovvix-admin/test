from django.shortcuts import render, redirect
from .models import UploadedImage
from django.conf import settings
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load YOLOv8 model
model_path = os.path.join(settings.BASE_DIR, 'best.pt')
model = YOLO(model_path)
print("Model loaded successfully!")

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Save uploaded image
        uploaded_image = UploadedImage(image=request.FILES['image'])
        uploaded_image.save()

        # Perform detection
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.image.name)
        processed_image, detected_objects = detect_objects(image_path)

        if processed_image is None:
            return render(request, 'detection/error.html', {'message': 'Error processing image'})

        # Save results
        results_path = os.path.join(settings.MEDIA_ROOT, 'results', uploaded_image.image.name)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        cv2.imwrite(results_path, processed_image)
        print(f"Processed image saved to: {results_path}")

        return render(request, r'C:\Users\dev12\OneDrive\Documents\Model\yolo_detection\detection\templates\result.html', {
            'original_image': uploaded_image.image.url,
            'processed_image': os.path.join('results', uploaded_image.image.name),
            'detected_objects': detected_objects,  # Pass detected objects to the template
        })

    return render(request, r'C:\Users\dev12\OneDrive\Documents\Model\yolo_detection\detection\templates\upload.html')

def detect_objects(image_path):
    # Load image
    try:
        image = Image.open(image_path)
        print(f"Image loaded successfully from: {image_path}")
        image = np.array(image)  # Convert to numpy array for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, []

    # Perform detection
    results = model(image)  # Run YOLOv8 inference
    print(f"Detection results: {results}")

    # Parse results
    detected_objects = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

        # Collect detected objects
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]
            detected_objects.append({
                'label': label,
                'confidence': float(conf),
                'box': [x1, y1, x2, y2],
            })

            # Draw bounding boxes on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image, detected_objects
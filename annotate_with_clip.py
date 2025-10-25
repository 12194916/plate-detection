from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import torch
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = 0.7  # Plate detection confidence
MIN_PLATE_AREA = 2000  # Minimum area for valid plate
MAX_DETECTIONS = 2  # Maximum plates per image
INTERIOR_THRESHOLD = 0.65  # CLIP score threshold for interior classification (increased for stricter filtering)
MIN_ASPECT_RATIO = 1.8  # License plates are typically 2-5x wider than tall
MAX_ASPECT_RATIO = 6.0  # Maximum width/height ratio
BLUR_INTENSITY = 60 # Blur strength for privacy (higher = more blur)

# Load CLIP model for interior/exterior classification
print("Loading CLIP model for interior/exterior classification...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load YOLOS for plate detection
print("Loading YOLOS model for plate detection...")
plate_detector = pipeline("object-detection",
                         model="nickmuchi/yolos-small-finetuned-license-plate-detection",
                         device=0)

def classify_interior_exterior(image):
    """
    Use CLIP to classify if image is interior or exterior view.
    Returns: 'interior' or 'exterior' and confidence score
    """
    # Define text prompts
    text_prompts = [
        "a photo taken from inside a car showing the dashboard and windshield",
        "a photo of the exterior of a car with license plate"
    ]

    inputs = clip_processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # probs[0][0] = interior probability, probs[0][1] = exterior probability
    interior_prob = probs[0][0].item()
    exterior_prob = probs[0][1].item()

    if interior_prob > INTERIOR_THRESHOLD:
        return 'interior', interior_prob
    else:
        return 'exterior', exterior_prob

def detect_plates(image):
    """
    Detect license plates in the image with filtering.
    """
    predictions = plate_detector(image)
    img_width, img_height = image.size

    filtered_predictions = []
    for pred in predictions:
        score = pred['score']
        box = pred['box']

        # Calculate dimensions
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        area = width * height

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0

        # Filter by confidence, area, and aspect ratio
        # License plates are wider than tall (typically 2-5x)
        if (score >= CONFIDENCE_THRESHOLD and
            area >= MIN_PLATE_AREA and
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            filtered_predictions.append(pred)

    # Sort by confidence and keep top detections
    filtered_predictions = sorted(filtered_predictions, key=lambda x: x['score'], reverse=True)[:MAX_DETECTIONS]

    return filtered_predictions

# Setup folders
data_folder = Path("data")
output_folder = Path("outputs")
interior_folder = output_folder / "interior"
exterior_folder = output_folder / "exterior"

output_folder.mkdir(exist_ok=True)
interior_folder.mkdir(exist_ok=True)
exterior_folder.mkdir(exist_ok=True)

# Process images
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(data_folder.rglob(ext))

print(f"\nProcessing {len(image_files)} images...")
print("=" * 70)

stats = {'interior': 0, 'exterior': 0, 'plates_detected': 0}

for img_path in image_files:
    # Load image
    image = Image.open(img_path)

    # Step 1: Classify interior vs exterior
    view_type, confidence = classify_interior_exterior(image)

    if view_type == 'interior':
        # Skip plate detection for interior images
        stats['interior'] += 1
        output_path = interior_folder / img_path.name
        image.save(output_path)
        print(f"🏠 INTERIOR ({confidence:.1%}): {img_path.name} -> Skipped plate detection")

    else:
        # Exterior image - run plate detection
        stats['exterior'] += 1
        predictions = detect_plates(image)

        if predictions:
            # Blur the license plate areas first
            for pred in predictions:
                box = pred['box']

                # Extract coordinates
                x1, y1 = int(box['xmin']), int(box['ymin'])
                x2, y2 = int(box['xmax']), int(box['ymax'])

                # Crop the plate region
                plate_region = image.crop((x1, y1, x2, y2))

                # Apply Gaussian blur
                blurred_plate = plate_region.filter(ImageFilter.GaussianBlur(BLUR_INTENSITY))

                # Paste the blurred region back
                image.paste(blurred_plate, (x1, y1))

            # Draw bounding boxes on top of blurred plates
            draw = ImageDraw.Draw(image)
            for pred in predictions:
                box = pred['box']
                score = pred['score']

                draw.rectangle(
                    [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
                    outline="red",
                    width=3
                )
                draw.text((box['xmin'], box['ymin'] - 20), f"{score:.2%}", fill="red")

            stats['plates_detected'] += len(predictions)
            output_path = exterior_folder / img_path.name
            confidences = [f"{p['score']:.1%}" for p in predictions]
            print(f"🚗 EXTERIOR ({confidence:.1%}): {img_path.name} -> {len(predictions)} plate(s) {confidences}")
        else:
            output_path = exterior_folder / f"no-plate_{img_path.name}"
            print(f"🚗 EXTERIOR ({confidence:.1%}): {img_path.name} -> No plates detected")

        image.save(output_path)

print("=" * 70)
print(f"\n✓ Processing Complete!")
print(f"  Interior images: {stats['interior']} (saved to outputs/interior/)")
print(f"  Exterior images: {stats['exterior']} (saved to outputs/exterior/)")
print(f"  Plates detected: {stats['plates_detected']}")

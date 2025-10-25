from transformers import pipeline
from PIL import Image, ImageDraw
from pathlib import Path

# Configuration
CONFIDENCE_THRESHOLD = 0.9  # Only keep detections above 90% confidence
MIN_PLATE_AREA = 3000  # Minimum area (pixels) for a valid plate (increased to filter small/distant plates)
MAX_DETECTIONS = 2  # Maximum number of plates per image
MIN_ASPECT_RATIO = 1.5  # License plates are wider than tall (typical: 2-5)
MAX_ASPECT_RATIO = 6.0  # Maximum width/height ratio
EXCLUDE_TOP_PERCENTAGE = 0.35  # Exclude detections in top 35% of image (mirrors, reflections)

# Load model with GPU
pipe = pipeline("object-detection", model="nickmuchi/yolos-small-finetuned-license-plate-detection", device=0)

# Setup folders
data_folder = Path("data")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# Process all images
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(data_folder.rglob(ext))

print(f"Processing {len(image_files)} images with confidence threshold: {CONFIDENCE_THRESHOLD}")
print("-" * 60)

for img_path in image_files:
    # Load image
    image = Image.open(img_path)
    img_width, img_height = image.size
    predictions = pipe(image)

    # Filter predictions by confidence threshold and area
    filtered_predictions = []
    for pred in predictions:
        score = pred['score']
        box = pred['box']

        # Calculate box dimensions
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        area = width * height

        # Calculate aspect ratio (width/height)
        aspect_ratio = width / height if height > 0 else 0

        # Calculate vertical position (exclude top portion - likely mirrors/reflections)
        box_center_y = (box['ymin'] + box['ymax']) / 2
        vertical_position = box_center_y / img_height  # 0 = top, 1 = bottom

        # Apply multiple filters
        if (score >= CONFIDENCE_THRESHOLD and
            area >= MIN_PLATE_AREA and
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and
            vertical_position > EXCLUDE_TOP_PERCENTAGE):
            filtered_predictions.append(pred)

    # Sort by confidence and keep top detections
    filtered_predictions = sorted(filtered_predictions, key=lambda x: x['score'], reverse=True)[:MAX_DETECTIONS]

    # Draw boxes if plates found
    if filtered_predictions:
        draw = ImageDraw.Draw(image)
        for pred in filtered_predictions:
            box = pred['box']
            score = pred['score']

            # Draw rectangle
            draw.rectangle(
                [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
                outline="red",
                width=3
            )

            # Draw confidence score
            text = f"{score:.2%}"
            draw.text((box['xmin'], box['ymin'] - 20), text, fill="red")

        output_path = output_folder / img_path.name
        confidences = [f"{p['score']:.2%}" for p in filtered_predictions]
        print(f"✓ {img_path.name}: Found {len(filtered_predictions)} plate(s) - Confidence: {confidences}")
    else:
        # No plate found - add tag
        output_path = output_folder / f"no-plate_{img_path.name}"
        print(f"✗ {img_path.name}: No plates detected (threshold: {CONFIDENCE_THRESHOLD})")

    # Save
    image.save(output_path)

print("-" * 60)
print(f"✓ Completed! Processed {len(image_files)} images. Saved to outputs/")



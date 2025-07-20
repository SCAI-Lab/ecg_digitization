import os
import cv2
import numpy as np
import random
from PIL import Image
import albumentations as A
from tqdm import tqdm

def get_polygon_mask(image):
    """
    Extracts the largest external polygon mask from an RGBA or BGR image.

    Args:
        image (np.ndarray): Input image in RGBA or BGR format.

    Returns:
        np.ndarray: An Nx2 array of (x, y) coordinates representing the largest polygon.
                    Returns an empty array if no valid contour is found.
    """
    
    alpha = image[:, :, 3] if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return largest[:, 0, :]
    
    return np.array([])

def polygon_to_yolo_seg(polygon, image_shape):
    """
    Converts polygon coordinates to YOLO segmentation format (normalized).

    Args:
        polygon (np.ndarray): Nx2 array of absolute (x, y) coordinates.
        image_shape (tuple): Shape of the image (height, width, channels).

    Returns:
        list of tuple: Normalized (x, y) coordinates for YOLO segmentation.
    """

    h, w = image_shape[:2]
    return [(x / w, y / h) for x, y in polygon]

def save_yolo_segmentation(file_path, class_id, yolo_polygon):
    """
    Saves a single polygon as a YOLO segmentation label.

    Args:
        file_path (str): Path to the label file to write to.
        class_id (int): Class index (usually 0 for binary classification).
        yolo_polygon (list): List of (x, y) normalized coordinates.
    """

    with open(file_path, "a") as f:
        poly_flat = " ".join([f"{x} {y}" for x, y in yolo_polygon])
        f.write(f"{class_id} {poly_flat}\n")

def augment_pulses(input_dir, output_image_dir, output_label_dir, num_augmentations=10):
    """
    Applies geometric and quality augmentations to reference pulse images and saves 
    the resulting images and YOLO-style polygon segmentation labels.

    Args:
        input_dir (str): Directory containing original pulse images.
        output_image_dir (str): Directory to save augmented images.
        output_label_dir (str): Directory to save YOLO segmentation labels.
        num_augmentations (int): Number of augmentations to perform per input image.
    """
    
    affine_transform = A.Affine(
        scale=(0.7, 0.95), rotate=(-10, 10), shear=(-5, 5), p=0.8
    )
    dropout_transform = A.CoarseDropout(
        num_holes_range=(1, 2),
        hole_height_range=(0.07, 0.12),
        hole_width_range=(0.07, 0.12),
        p=0.4
    )
    
    pulse_quality_aug = A.Compose([
        A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
        A.GaussianBlur(sigma_limit=(0.2, 0.5), blur_limit=0, p=0.2)
    ])

    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue

        base_name = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            continue

        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)

        for j in range(num_augmentations):
            orig_poly = get_polygon_mask(image)
            
            if orig_poly.size == 0:
                continue

            transformed = affine_transform(image=image)
            transformed_image = transformed["image"]

            transformed_poly = get_polygon_mask(transformed_image)
            
            if transformed_poly.size == 0:
                continue

            yolo_polygon = polygon_to_yolo_seg(transformed_poly, transformed_image.shape)
            label_path = os.path.join(output_label_dir, f"{base_name}_{j}.txt")
            save_yolo_segmentation(label_path, class_id=0, yolo_polygon=yolo_polygon)

            visual_augmented = pulse_quality_aug(image=transformed_image)["image"]
            final_image = dropout_transform(image=visual_augmented)["image"]

            output_path = os.path.join(output_image_dir, f"{base_name}_{j}.png")
            Image.fromarray(final_image).save(output_path)

def load_yolo_polygon(label_path, image_shape):
    """
    Loads a YOLO polygon label and converts it back to pixel coordinates.

    Args:
        label_path (str): Path to the YOLO label file.
        image_shape (tuple): Shape of the image used to denormalize coordinates (height, width).

    Returns:
        np.ndarray: An Nx2 array of absolute (x, y) polygon coordinates.
    """

    h, w = image_shape[:2]
    with open(label_path, 'r') as f:
        parts = f.readline().strip().split()
        coords = list(map(float, parts[1:]))
        polygon = [(x * w, y * h) for x, y in zip(coords[::2], coords[1::2])]
        return np.array(polygon, dtype=np.float32)

def boxes_overlap(box1, box2):
    """
    Checks whether two bounding boxes overlap.

    Args:
        box1 (tuple): (x_min, y_min, x_max, y_max) for the first box.
        box2 (tuple): (x_min, y_min, x_max, y_max) for the second box.

    Returns:
        bool: True if boxes overlap, False otherwise.
    """

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

def overlay_random_pulses_on_ecgs(ecg_dir, pulse_image_dir, pulse_label_dir, output_image_dir, output_label_dir,
                                  min_pulses=3, max_pulses=13, margin=128, num_repeats_per_ecg=3, max_attempts_per_pulse=20):
    """
    Randomly overlays augmented pulse images onto ECG images without overlap,
    and saves the final composited image and corresponding YOLO-style segmentation labels.

    Args:
        ecg_dir (str): Directory containing background ECG images.
        pulse_image_dir (str): Directory containing augmented pulse images.
        pulse_label_dir (str): Directory containing YOLO polygon labels for pulses.
        output_image_dir (str): Output directory for composited images.
        output_label_dir (str): Output directory for generated YOLO labels.
        min_pulses (int): Minimum number of pulses to place per ECG.
        max_pulses (int): Maximum number of pulses to place per ECG.
        margin (int): Margin in pixels to keep around placed pulses.
        num_repeats_per_ecg (int): Number of augmented versions to generate per ECG.
        max_attempts_per_pulse (int): Max placement attempts per pulse before skipping.
    """
    
    pulse_images = sorted([f for f in os.listdir(pulse_image_dir) if f.endswith(".png")])
    pulse_labels = sorted([f for f in os.listdir(pulse_label_dir) if f.endswith(".txt")])
    pulse_pairs = list(zip(pulse_images, pulse_labels))

    ecg_filenames = [f for f in os.listdir(ecg_dir) if f.endswith(".png")]

    for ecg_file in tqdm(ecg_filenames, desc="Processing ECGs"):
        ecg_path = os.path.join(ecg_dir, ecg_file)

        for repeat_idx in tqdm(range(num_repeats_per_ecg), desc=f"Repeats for {ecg_file}", leave=False):
            ecg = cv2.imread(ecg_path, cv2.IMREAD_COLOR)
            if ecg is None:
                continue
            
            target_height = 1700
            original_height, original_width = ecg.shape[:2]
            aspect_ratio = original_width / original_height
            new_width = int(target_height * aspect_ratio)

            if original_height > target_height:
                resized_ecg = cv2.resize(ecg, (new_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                resized_ecg = cv2.resize(ecg, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
            
            ecg_pil = Image.fromarray(cv2.cvtColor(resized_ecg, cv2.COLOR_BGR2RGB)).convert("RGBA")
         
            ew, eh = ecg_pil.size
            all_yolo_polygons = []
            placed_boxes = []

            n_pulses = random.randint(min_pulses, max_pulses)
            chosen_pulses = random.choices(pulse_pairs, k=n_pulses)

            for img_name, lbl_name in chosen_pulses:
                pulse_rgba = Image.open(os.path.join(pulse_image_dir, img_name)).convert("RGBA")
                pw, ph = pulse_rgba.size

                if pw + 2 * margin > ew or ph + 2 * margin > eh:
                    continue

                placed = False
                for _ in range(max_attempts_per_pulse):
                    offset_x = random.randint(margin, ew - pw - margin)
                    offset_y = random.randint(margin, eh - ph - margin)
                    new_box = (offset_x, offset_y, offset_x + pw, offset_y + ph)

                    if any(boxes_overlap(new_box, b) for b in placed_boxes):
                        continue

                    ecg_pil.paste(pulse_rgba, (offset_x, offset_y), mask=pulse_rgba)
                    placed_boxes.append(new_box)

                    polygon = load_yolo_polygon(os.path.join(pulse_label_dir, lbl_name), (ph, pw))
                    shifted_polygon = polygon + np.array([offset_x, offset_y])
                    yolo_polygon = polygon_to_yolo_seg(shifted_polygon, (eh, ew))
                    all_yolo_polygons.append((0, yolo_polygon))

                    placed = True
                    break 

                if not placed:
                    print(f"Skipped pulse {img_name} due to overlap.")

            base_name = os.path.splitext(ecg_file)[0]
            out_img_name = f"{base_name}_{repeat_idx}.png"
            result_path = os.path.join(output_image_dir, out_img_name)
            ecg_result = ecg_pil.convert("RGB")
            ecg_result.save(result_path)

            label_output_path = os.path.join(output_label_dir, out_img_name.replace(".png", ".txt"))
            with open(label_output_path, "w") as f:
                for class_id, yolo_polygon in all_yolo_polygons:
                    poly_flat = " ".join([f"{x} {y}" for x, y in yolo_polygon])
                    f.write(f"{class_id} {poly_flat}\n")

def main():
    # Directories
    ref_pulse_dir = "pulses/"
    ecg_dir = "images_without_pulse/"
    augmented_pulse_dir = "datasets/data_pulse/ref_pulse_augment"
    augmented_label_dir = "datasets/data_pulse/ref_pulse_augment_labels"
    output_ecg_dir = "datasets/data_pulse/images/train"
    output_label_dir = "datasets/data_pulse/labels/train"

    # Make directories if they don't exist
    os.makedirs(augmented_pulse_dir, exist_ok=True)
    os.makedirs(augmented_label_dir, exist_ok=True)
    os.makedirs(output_ecg_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Generate augmented reference pulses and save the images and labels
    augment_pulses(ref_pulse_dir, augmented_pulse_dir, augmented_label_dir, num_augmentations=10)

    # Overlay the augmented reference pulses onto ECG images
    overlay_random_pulses_on_ecgs(
        ecg_dir=ecg_dir,
        pulse_image_dir=augmented_pulse_dir,
        pulse_label_dir=augmented_label_dir,
        output_image_dir=output_ecg_dir,
        output_label_dir=output_label_dir,
        num_repeats_per_ecg=1
    )

if __name__ == "__main__":
    main()

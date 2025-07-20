import os
from tqdm import tqdm
import shutil
import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def move_files_by_extension(source_dir, dest_dir, file_extension):
    """
    Move all files with a specific extension from one directory to another.

    Args:
        source_dir (str): Path to the directory containing source files.
        dest_dir (str): Path to the directory where files should be moved.
        file_extension (str): File extension to filter by (e.g., '.png', '.json').

    Raises:
        ValueError: If the source directory does not exist.
    """
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)
    
    files_to_move = [f for f in os.listdir(source_dir) if f.endswith(file_extension)]

    for filename in tqdm(files_to_move, desc=f"Moving '{file_extension}' files"):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)

    print(f"Moved {len(files_to_move)} '{file_extension}' files from {source_dir} to {dest_dir}")

# ----- Functions to generate annotations from ECG-Image-Kit generated images -----

def extract_contours(signal, img_width, img_height, upsample_factor=10):
    """
    Extract contours from a signal by drawing it on a mask and finding boundaries.

    Args:
        signal (list): List of (y, x) tuples representing points in the signal.
        img_width (int): Width of the target image.
        img_height (int): Height of the target image.
        upsample_factor (int): Factor to scale up coordinates for better resolution.

    Returns:
        list: List of contours found in the mask.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(len(signal) - 1):
        y1, x1 = signal[i]
        y2, x2 = signal[i + 1]
        
        x1 *= upsample_factor
        y1 *= upsample_factor
        x2 *= upsample_factor
        y2 *= upsample_factor
        
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2*upsample_factor)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def write_annotations(file_path, leads, img_width, img_height, upsample_factor=10):
    """
    Write YOLO-style annotations from signal data to a text file.

    Args:
        file_path (str): Output text file path.
        leads (list): List of lead dictionaries with 'plotted_pixels' keys.
        img_width (int): Width of the (upsampled) image.
        img_height (int): Height of the (upsampled) image.
        upsample_factor (int): Upsampling factor applied to original dimensions.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:

            for lead in leads:
                signal = np.array(lead['plotted_pixels'])
            
                contours = extract_contours(signal, img_width, img_height, upsample_factor)
                
                for contour in contours:
                    f.write("0 ")  # Assuming "0" is a label or identifier
                    for point in contour:
                        x, y = point[0]  # Contours are arrays of points
                        f.write(f"{x / img_width} {y / img_height} ")
                    f.write("\n")

    except Exception as e:
        print(f"Error writing annotations: {e}")

def generate_annotations(file_path, txt_file_path, upsample_factor=10):
    """
    Generate YOLO annotations from a JSON file and write to a text file.

    Args:
        file_path (str): Path to the JSON file.
        txt_file_path (str): Destination text file path for annotations.
        upsample_factor (int): Upsampling factor applied to width/height.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        leads = data.get('leads', [])
        img_width = data.get('width')
        img_height = data.get('height')
        
        if not leads or img_width is None or img_height is None:
            raise KeyError("Missing 'leads', 'width', or 'height' in JSON data.")
        
        upsampled_height = img_height * upsample_factor
        upsampled_width = img_width * upsample_factor
        
        write_annotations(txt_file_path, leads, upsampled_width, upsampled_height)
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except KeyError as e:
        print(f"Error: Missing key in JSON data - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def annotate_images(src_folder, dest_folder):
    """
    Generate annotation files from JSON files in the source folder.

    Args:
        src_folder (str): Folder containing input JSON files.
        dest_folder (str): Folder where .txt annotation files will be saved.
    """
    os.makedirs(dest_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        file_stem = os.path.splitext(filename)[0]
        file_stem_txt = file_stem + '.txt'
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dest_folder, file_stem_txt)
        generate_annotations(src_path, dst_path)

# ----- Functions to plot images and annotations -----
        
def plot_annotations(img_path, annotations_path, save_file=False, save_path=None):
    """
    Plot YOLO-style polygon annotations on an image.

    Args:
        img_path (str): Path to the image file.
        annotations_path (str): Path to the .txt annotation file.
        save_file (bool): Whether to save the image with drawn annotations.
        save_path (str): Path to save the annotated image if save_file is True.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        with open(annotations_path, 'r', encoding='utf-8') as file:
            annotations = file.readlines()

        for annotation in annotations:
            values = annotation.strip().split()
            if len(values) < 3 or (len(values) - 1) % 2 != 0:
                print(f"Warning: Invalid annotation format in line - {annotation}")
                continue

            # Extract points from annotation
            points = [(float(values[i]), float(values[i + 1])) for i in range(1, len(values), 2)]
            points = [(int(x * img.shape[1]), int(y * img.shape[0])) for x, y in points]
            points = np.array(points, np.int32).reshape((-1, 1, 2))

            # Generate a random color for each annotation
            color = np.random.randint(0, 256, 3).tolist()  # Random RGB color

            # Draw the polygon with the generated random color
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        # Convert BGR image to RGB for correct display in matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Optional: Hide axes for cleaner view
        plt.show()

        if save_file and save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            print(f"Image with annotations saved to: {save_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def draw_yolo_boxes(image_path, label_path, class_names=None):
    """
    Draw YOLO bounding boxes on an image using annotations from a label file.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO annotation file (.txt).
        class_names (list, optional): List of class names to map class IDs to labels.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    if not os.path.exists(label_path):
        print(f"No label file found: {label_path}")
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines

            class_id, cx, cy, bw, bh = map(float, parts)
            cx *= w
            cy *= h
            bw *= w
            bh *= h

            x0 = cx - bw / 2
            y0 = cy - bh / 2
            x1 = cx + bw / 2
            y1 = cy + bh / 2

            draw.rectangle([x0, y0, x1, y1], outline="red", width=1)

            if class_names:
                class_id = int(class_id)
                label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                draw.text((x0, y0 - 10), label, fill="red")

    plt.imshow(image)
    plt.axis("off")
    plt.show()

def visualize_image_grid(folder_path, prefix="xxx"):
    """
    Visualize a grid of images based on filename patterns like 'xxx_row_col.png'.

    Args:
        folder_path (str): Path to the folder containing image tiles.
        prefix (str): Prefix of image filenames.
    """
    images = []

    for fname in os.listdir(folder_path):
        if fname.startswith(prefix) and fname.endswith(".png"):
            try:
                parts = fname[:-4].split('_')  # remove ".png" and split by "_"
                row = int(parts[-2])
                col = int(parts[-1])
                images.append((row, col, fname))
            except (IndexError, ValueError):
                continue

    if not images:
        print("No matching images found.")
        return

    max_row = max(row for row, _, _ in images)
    max_col = max(col for _, col, _ in images)

    # Create empty grid
    grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for row, col, fname in images:
        img = Image.open(os.path.join(folder_path, fname))
        grid[row][col] = img

    # Create subplots
    fig, axes = plt.subplots(nrows=max_row + 1, ncols=max_col + 1, figsize=(3 * (max_col + 1), 3 * (max_row + 1)))

    for r in range(max_row + 1):
        for c in range(max_col + 1):
            ax = axes[r][c] if max_row > 0 and max_col > 0 else axes[max(c, r)]  # handle 1D case
            ax.axis('off')

            img = grid[r][c]
            if img is not None:
                ax.imshow(img)

    plt.tight_layout()
    plt.show()

# ----- Functions to rearrange splits -----

def split_files(folder_im, folder_la, val_ratio=0.2, test_ratio=0.0, seed=None):
    """
    Split training images and labels into 'val' and 'test' subsets.

    Args:
        folder_im (str): Full path to the image folder (contains 'train').
        folder_la (str): Full path to the label folder (contains 'train').
        val_ratio (float): Proportion of files to allocate to validation.
        test_ratio (float): Proportion of files to allocate to testing.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Define subdirectories
    train_img_dir = os.path.join(folder_im, 'train')
    val_img_dir   = os.path.join(folder_im, 'val')
    test_img_dir  = os.path.join(folder_im, 'test')

    train_label_dir = os.path.join(folder_la, 'train')
    val_label_dir   = os.path.join(folder_la, 'val')
    test_label_dir  = os.path.join(folder_la, 'test')

    # Ensure destination directories exist
    for d in [val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        os.makedirs(d, exist_ok=True)

    # List all image files
    train_img_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    total = len(train_img_files)

    num_val = int(val_ratio * total)
    num_test = int(test_ratio * total)

    val_files = random.sample(train_img_files, num_val)
    remaining = list(set(train_img_files) - set(val_files))
    test_files = random.sample(remaining, num_test)

    def move_files(files, src_img, dst_img, src_lab, dst_lab):
        for fname in tqdm(files, desc=f"Moving to {os.path.basename(dst_img)}"):
            shutil.move(os.path.join(src_img, fname), os.path.join(dst_img, fname))
            label = os.path.splitext(fname)[0] + '.txt'
            shutil.move(os.path.join(src_lab, label), os.path.join(dst_lab, label))

    move_files(val_files, train_img_dir, val_img_dir, train_label_dir, val_label_dir)
    move_files(test_files, train_img_dir, test_img_dir, train_label_dir, test_label_dir)

def move_split_files(folder_im, folder_la, from_splits=('val', 'test'), to_split='train'):
    """
    Move images and label files from 'val' and 'test' to the 'train' folder.

    Args:
        folder_im (str): Full path to the image folder root.
        folder_la (str): Full path to the label folder root.
        from_splits (tuple): Subfolders to move from (default: ('val', 'test')).
        to_split (str): Target subfolder to move files to (default: 'train').
    """
    def move_all(src_dir, dst_dir):
        if not os.path.exists(src_dir):
            print(f"Source directory does not exist: {src_dir}")
            return
        os.makedirs(dst_dir, exist_ok=True)
        for filename in tqdm(os.listdir(src_dir), desc=f"Moving from {src_dir}"):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.move(src_path, dst_path)

    for split in from_splits:
        # Image files
        im_src = os.path.join(folder_im, split)
        im_dst = os.path.join(folder_im, to_split)
        move_all(im_src, im_dst)

        # Label files
        la_src = os.path.join(folder_la, split)
        la_dst = os.path.join(folder_la, to_split)
        move_all(la_src, la_dst)

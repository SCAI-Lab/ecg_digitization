import numpy as np
import os
import cv2
import json
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def extract_contours(signal, img_width, img_height, upsample_factor_width, upsample_factor_height, patch_size):
    """
    Draws the signal as contours on a blank image and extracts image patches.

    Args:
        signal (np.ndarray): Array of (y, x) plotted pixel coordinates.
        img_width (int): Width of the target image.
        img_height (int): Height of the target image.
        upsample_factor_width (float): Scaling factor for width.
        upsample_factor_height (float): Scaling factor for height.
        patch_size (int): Size of each patch to extract.

    Returns:
        np.ndarray: 2D array of image patches with contours.
    """

    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(len(signal) - 1):
        y1, x1 = signal[i]
        y2, x2 = signal[i + 1]
        
        x1 *= upsample_factor_width
        y1 *= upsample_factor_height
        x2 *= upsample_factor_width
        y2 *= upsample_factor_height

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        cv2.line(mask, (x1, y1), (x2, y2), 255, int((29 / 4000) * patch_size))
    
    return patch_image(mask, img_width, img_height, patch_size)

def write_annotations(file_path, leads, base_name, img_width, img_height, upsample_factor_width, upsample_factor_height, patch_size):
    """
    Converts leads into contour annotations and writes them to YOLO label files.

    Args:
        file_path (str): Directory to save the annotation files.
        leads (list): List of lead dictionaries with 'plotted_pixels'.
        base_name (str): Base name for saving.
        img_width (int): Target width after upsampling.
        img_height (int): Target height after upsampling.
        upsample_factor_width (float): Width scaling factor.
        upsample_factor_height (float): Height scaling factor.
        patch_size (int): Size of each image patch.
    """

    for lead in leads:
        signal = np.array(lead['plotted_pixels'])
        
        patches_img = extract_contours(signal, img_width, img_height, upsample_factor_width, upsample_factor_height, patch_size)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                contours, _ = cv2.findContours(patches_img[i][j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                with open(f"{file_path}/{base_name}_{i}_{j}.txt", "a") as f:
                    for contour in contours:
                        f.write("0 ")  
                        for point in contour:
                            x, y = point[0] 
                            f.write(f"{x / patches_img[i][j].shape[1]} {y / patches_img[i][j].shape[0]} ")
                        f.write("\n")

def extract_grid_patches(image_np, patch_size, stride, num_patches):
    """
    Extracts a grid of patches from the input image.

    Args:
        image_np (np.ndarray): Input image as numpy array.
        patch_size (int): Size of each patch (square).
        stride (Tuple[int, int]): Stride in (x, y) directions.
        num_patches (Tuple[int, int]): Number of patches in (x, y).

    Returns:
        np.ndarray: 2D array of image patches.
    """
    
    patches = []
    
    for i in range(0, num_patches[1]+1):
        row_patches = []
        for j in range(0, num_patches[0]+1):
            patch = image_np[i*stride[1]:i*stride[1] + patch_size, j*stride[0]:j*stride[0] + patch_size]
            row_patches.append(patch)
        patches.append(row_patches)

    return np.array(patches)

def save_patches(patches_img, out_dir, base_name):
    """
    Saves image patches to the specified directory.

    Args:
        patches_img (np.ndarray): 2D array of image patches.
        out_dir (str): Output directory for patches.
        base_name (str): Base name for saving patches.
    """

    os.makedirs(out_dir, exist_ok=True)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            img_patch = Image.fromarray(patches_img[i, j])
            img_patch.save(os.path.join(out_dir, f"{base_name}_{i}_{j}.png"))

def patch_image(image_np, img_width, img_height, patch_size):
    """
    Divides an image into a grid of overlapping patches.

    Args:
        image_np (np.ndarray): Input image.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        patch_size (int): Size of each patch.

    Returns:
        np.ndarray: 2D array of image patches.
    """

    num_patches_x = (img_width // patch_size)
    num_patches_y = (img_height // patch_size)

    stride_x = (img_width - patch_size) // num_patches_x
    stride_y = (img_height - patch_size) // num_patches_y

    patches_img = extract_grid_patches(image_np, patch_size, (stride_x, stride_y), (num_patches_x, num_patches_y))

    return patches_img

def visualize_patches(patches_img, base_name):
    """
    Visualizes image patches in a grid using matplotlib.

    Args:
        patches_img (np.ndarray): 2D array of patches to visualize.
        base_name (str): Title for the figure.
    """

    num_rows, num_cols = patches_img.shape[:2]
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(f'Patches for {base_name}', fontsize=16)

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i][j] if num_rows > 1 and num_cols > 1 else (axs[i] if num_cols == 1 else axs[j])
            ax.imshow(patches_img[i, j], cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def generate_annotations(image_path, out_dir, base_name, file_path, txt_file_path, patch_size):
    """
    Generates and saves patches and annotation files for one ECG image.

    Args:
        image_path (str): Path to the ECG image.
        out_dir (str): Directory to save image patches.
        base_name (str): Base filename for outputs.
        file_path (str): Path to the corresponding JSON annotation file.
        txt_file_path (str): Output directory for label files.
        patch_size (int): Size of each patch.
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    leads = data.get('leads', [])
    img_width = data.get('width')
    img_height = data.get('height')
    
    if not leads or img_width is None or img_height is None:
        raise KeyError("Missing 'leads', 'width', or 'height' in JSON data.")
    
    upsampled_height = 22000
    upsampled_width = 17000

    upsample_factor_height = upsampled_height / img_height
    upsample_factor_width = upsampled_width / img_width

    write_annotations(txt_file_path, 
                      leads, 
                      base_name, 
                      upsampled_width, 
                      upsampled_height, 
                      upsample_factor_width,
                      upsample_factor_height, 
                      patch_size)

    image = cv2.imread(image_path)
    
    upsampled_image = cv2.resize(image, (upsampled_width, upsampled_height), interpolation=cv2.INTER_CUBIC)

    patches_img = patch_image(upsampled_image, upsampled_width, upsampled_height, patch_size)

    save_patches(patches_img, out_dir, base_name)

def process_single_image(image_file_args):
    """
    Worker function to process a single image and generate its annotations.

    Args:
        image_file_args (Tuple): Contains image file name and all processing parameters.

    Returns:
        None
    """

    image_file, image_folder, json_folder, out_dir, txt_file_path, patch_size = image_file_args
    
    image_path = os.path.join(image_folder, image_file)
    json_file_name = image_file.replace('.png', '.json')
    json_path = os.path.join(json_folder, json_file_name)

    if os.path.exists(json_path):
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        generate_annotations(image_path, out_dir, base_name, json_path, txt_file_path, patch_size)
    else:
        print(f"Warning: JSON file not found for {image_file}")

def main():

    # Directories
    image_folder = "datasets/data_full/images/train/" # Directory with full ECG images
    json_folder = "datasets/data_full/json/" # Directory with full ECG images label files
    out_dir = "datasets/data_patch/images/train" # Directory to save patched images
    txt_dir = "datasets/data_patch/labels/train" # Directory to save patched images labels

    # Make directories if they don't exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # Parameters for the sizes of the patches and number of images to patch with the corresponding patch sizes
    patch_sizes = [3000, 5000, 4000]
    group_counts = [3, 3, 4]

    # Randomly sample images
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    selected_images = random.sample(image_files, min(sum(group_counts), len(image_files)))

    
    # Use a multiprocessing pool to generate patches per patch size group
    start = 0
    for group_idx, (count, patch_size) in enumerate(zip(group_counts, patch_sizes)):
        group_images = selected_images[start:start+count]
        start += count

        task_args = [(f, image_folder, json_folder, out_dir, txt_dir, patch_size) for f in group_images]

        print(f"Processing group {group_idx+1} with patch size {patch_size} on {len(task_args)} images...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_single_image, task_args), total=len(task_args), desc=f"Group {group_idx+1}"))

if __name__ == '__main__':
    main()





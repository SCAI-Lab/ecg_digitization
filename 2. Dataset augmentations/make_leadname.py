import os
import random
from PIL import Image, ImageDraw, ImageFont
import glob
import random
import json
from tqdm import tqdm
import concurrent.futures
import math

def load_fonts(font_dir):
    """
    Load all .ttf and .otf fonts from the specified directory.

    Args:
        font_dir (str): Directory containing font files.

    Returns:
        List[str]: List of font file paths.
    """
    
    font_paths = glob.glob(os.path.join(font_dir, "*.ttf")) + glob.glob(os.path.join(font_dir, "*.otf"))
    return font_paths

def add_rotated_text(image, text, position, font, angle):
    """
    Draw rotated text onto the image.

    Args:
        image (PIL.Image): Base image.
        text (str): Text to draw.
        position (Tuple[int, int]): (x, y) position for the text.
        font (PIL.ImageFont.FreeTypeFont): Font to use.
        angle (float): Rotation angle in degrees.

    Returns:
        PIL.Image: Image with the rotated text overlaid.
    """
    
    txt_img = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)
    draw.text(position, text, font=font, fill=(0, 0, 0, 255))
    txt_img = txt_img.rotate(angle, resample=Image.BICUBIC, center=position)
    return Image.alpha_composite(image.convert("RGBA"), txt_img)

def add_rotated_text_with_spacing(image, text, position, font, angle, spacing=0):
    """
    Draw rotated text with spacing between characters.

    Args:
        image (PIL.Image): Base image.
        text (str): Text to draw.
        position (Tuple[int, int]): (x, y) position.
        font (PIL.ImageFont.FreeTypeFont): Font to use.
        angle (float): Rotation angle in degrees.
        spacing (int): Additional space between characters.

    Returns:
        PIL.Image: Image with the spaced, rotated text.
    """
    
    x, y = position
    txt_img = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)

    for char in text:
        draw.text((x, y), char, font=font, fill=(0, 0, 0, 255))
        char_width = font.getbbox(char)[2]
        x += char_width + spacing

    txt_img = txt_img.rotate(angle, resample=Image.BICUBIC, center=position)
    return Image.alpha_composite(image.convert("RGBA"), txt_img)

def get_yolo_bbox(text, font, position, img_size, angle=0, spacing=0):
    """
    Compute YOLO-format bounding box for spaced, rotated text.

    Args:
        text (str): Text to compute the bounding box for.
        font (PIL.ImageFont.FreeTypeFont): Font used.
        position (Tuple[int, int]): Position of the text.
        img_size (Tuple[int, int]): Image dimensions (width, height).
        angle (float): Rotation angle in degrees.
        spacing (int): Space between characters.

    Returns:
        Tuple[float, float, float, float]: Normalized (x_center, y_center, width, height).
    """
    
    x, y = position
    txt_img = Image.new('RGBA', img_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)

    for char in text:
        draw.text((x, y), char, font=font, fill=(0, 0, 0, 255))
        char_width = font.getbbox(char)[2] - font.getbbox(char)[0]
        x += char_width + spacing

    rotated = txt_img.rotate(angle, resample=Image.BICUBIC, center=position)

    alpha = rotated.split()[-1]  
    bbox = alpha.getbbox()

    if bbox is None:
        return 0, 0, 0, 0 

    x0, y0, x1, y1 = bbox
    w, h = img_size
    center_x = (x0 + x1) / 2 / w
    center_y = (y0 + y1) / 2 / h
    width = ((3*w/600) + x1 - x0) / w
    height = ((3*h/500) + y1 - y0) / h

    return center_x, center_y, width, height


def append_label(label_path, class_id, bbox):
    """
    Append a single YOLO label line to a text file.

    Args:
        label_path (str): Path to the label file.
        class_id (int): Class ID for the label.
        bbox (Tuple[float, float, float, float]): Bounding box in YOLO format.
    """
    
    with open(label_path, "a") as f:
        f.write(f"{class_id} {' '.join(str(b) for b in bbox)}\n")


def write_lead_bbox(file_path, label_path, lead_name_to_id):
    """
    Convert JSON annotations into YOLO-format and write to file.

    Args:
        file_path (str): Path to the input JSON file.
        label_path (str): Output label file path.
        lead_name_to_id (dict): Mapping from lead names to class IDs.
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    leads = data.get('leads', [])
    img_width = data.get('width')
    img_height = data.get('height')
    
    with open(label_path, 'w') as label_file:
        for lead in leads:
            text_box = lead["text_bounding_box"]
            lead_name = lead["lead_name"]
    
            points = [
                (text_box['0'][1], text_box['0'][0]),
                (text_box['1'][1], text_box['1'][0]),
                (text_box['2'][1], text_box['2'][0]),
                (text_box['3'][1], text_box['3'][0])
            ]
    
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
    
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords) - 2
            max_y = max(y_coords) - 2
    
            box_w = (3*img_width/600) + max_x - min_x
            box_h = (3*img_height/500) + max_y - min_y
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
    
            center_x_norm = center_x / img_width
            center_y_norm = center_y / img_height
            width_norm = box_w / img_width
            height_norm = box_h / img_height
            
            lead_id = lead_name_to_id.get(normalize_lead_name(lead_name))

            label_file.write(f"{lead_id} {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n")

def get_safe_position(font, text, img_width, img_height, existing_bboxes, max_tries=100):
    """
    Get a safe random position for placing text, avoiding overlaps.

    Args:
        font (PIL.ImageFont.FreeTypeFont): Font to be used.
        text (str): Text to render.
        img_width (int): Image width.
        img_height (int): Image height.
        existing_bboxes (List[Tuple[int, int, int, int]]): List of existing bounding boxes.
        max_tries (int): Maximum number of placement attempts.

    Returns:
        Tuple[int, int]: Safe (x, y) position for the text.
    """
    
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    for _ in range(max_tries):
        tx = random.randint(0, max(0, img_width - text_width))
        ty = random.randint(0, max(0, img_height - text_height))

        new_bbox = (tx, ty, tx + text_width, ty + text_height)

        # Check for overlap
        overlap = any(
            not (new_bbox[2] < bbox[0] or new_bbox[0] > bbox[2] or
                 new_bbox[3] < bbox[1] or new_bbox[1] > bbox[3])
            for bbox in existing_bboxes
        )

        if not overlap:
            existing_bboxes.append(new_bbox)
            return tx, ty

    tx = random.randint(0, max(0, img_width - text_width))
    ty = random.randint(0, max(0, img_height - text_height))
    existing_bboxes.append((tx, ty, tx + text_width, ty + text_height))
    
    return tx, ty

def get_safe_position_with_spacing(font, text, img_width, img_height, existing_bboxes, spacing=0, margin_ratio=0.02, max_tries=100):
    """
    Get a safe position for spaced text, avoiding bounding box overlaps.

    Args:
        font (PIL.ImageFont.FreeTypeFont): Font object.
        text (str): Text string.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        existing_bboxes (List[Tuple[int, int, int, int]]): Already used bounding boxes.
        spacing (int): Inter-character spacing.
        margin_ratio (float): Margin around edges.
        max_tries (int): Retry attempts for a safe spot.

    Returns:
        Tuple[int, int]: Safe (x, y) coordinates.
    """
    
    text_width = sum((font.getbbox(c)[2] - font.getbbox(c)[0] + spacing) for c in text) - spacing
    text_height = font.getbbox(text)[3] - font.getbbox(text)[1]

    margin_x = int(margin_ratio * img_width)
    margin_y = int(margin_ratio * img_height)

    for _ in range(max_tries):
        tx = random.randint(math.ceil(margin_x), math.floor(img_width - text_width - margin_x))
        ty = random.randint(math.ceil(margin_y), math.floor(img_height - text_height - margin_y))

        new_bbox = (tx, ty, tx + text_width, ty + text_height)

        overlap = any(
            not (new_bbox[2] < bbox[0] or new_bbox[0] > bbox[2] or
                 new_bbox[3] < bbox[1] or new_bbox[1] > bbox[3])
            for bbox in existing_bboxes
        )

        if not overlap:
            existing_bboxes.append(new_bbox)
            return tx, ty

    tx = random.randint(margin_x, max(margin_x, img_width - text_width - margin_x))
    ty = random.randint(margin_y, max(margin_y, img_height - text_height - margin_y))
    existing_bboxes.append((tx, ty, tx + text_width, ty + text_height))

    return tx, ty

def normalize_lead_name(name):
    return name.upper().replace("|", "I").replace("C", "V")

def process_image(args):
    """
    Apply synthetic lead and distractor text to an image and save new image and label.

    Args:
        args (Tuple): 
            - img_path (str): Input image path.
            - font_paths (List[str]): List of available fonts.
            - lead_names (List[str]): Valid ECG lead names.
            - distractor_chars (List[str]): Non-lead distractor characters.
            - max_extra_leads (int): Max number of random lead text to add.
            - max_distractors (int): Max number of distractor characters.
            - label_dir (str): Directory for output YOLO label files.
            - json_dir (str): Directory for original JSON annotations.
            - img_lead_dir (str): Output image directory with synthetic text.
    """
    
    img_path, font_paths, lead_names, distractor_chars, max_extra_leads, max_distractors, label_dir, json_dir, img_lead_dir = args

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, img_name + ".txt")
    file_path = os.path.join(json_dir, img_name + ".json")

    lead_name_to_id = {
        "I": 0,
        "II": 1,
        "III": 2,
        "AVR": 3,
        "AVL": 4,
        "AVF": 5,
        "V1": 6,
        "V2": 7,
        "V3": 8,
        "V4": 9,
        "V5": 10,
        "V6": 11
    }
    
    write_lead_bbox(file_path, label_path, lead_name_to_id)

    image = Image.open(img_path).convert("RGBA")
    w, h = image.size
    
    placed_bboxes = []

    # Add lead names
    for _ in range(random.randint(1, max_extra_leads)):
        text = random.choice(lead_names)
        font_path = random.choice(font_paths)
        font_size = random.randint(int((25/1400)*h), int((30/1400)*h))
        font = ImageFont.truetype(font_path, font_size)
        angle = random.gauss(0, 5)
        angle = max(min(angle, 10), -10)
        spacing = int(random.gauss(0, font_size * 0.07))
        spacing = max(-(font_size * 0.15), min(spacing, font_size * 0.15)) 
        tx, ty = get_safe_position_with_spacing(font, text, w, h, placed_bboxes, spacing)

        image = add_rotated_text_with_spacing(image, text, (tx, ty), font, angle, spacing)
        bbox = get_yolo_bbox(text, font, (tx, ty), (w, h), angle, spacing)
        lead_class_id = lead_name_to_id.get(normalize_lead_name(text))
        append_label(label_path, lead_class_id, bbox)

    # Add other random letters
    for _ in range(random.randint(1, max_distractors)):
        text = random.choice(distractor_chars)
        font_path = random.choice(font_paths)
        font = ImageFont.truetype(font_path, random.randint(int((20/1400)*h), int((25/1400)*h)))
        angle = random.gauss(0, 5)
        angle = max(min(angle, 10), -10)        
        tx, ty = get_safe_position(font, text, w, h, placed_bboxes)
        image = add_rotated_text(image, text, (tx, ty), font, angle)

    image_lead_path = os.path.join(img_lead_dir, img_name + ".png")
    image.convert("RGB").save(image_lead_path)

def main():
    # List of all valid lead names
    lead_names = ["I", "II", "III", 
                  "I", "II", "III", 
                  "I", "II", "III", 
                  "|||", "||", "|", 
                  "aVR", "aVL", "aVF", 
                  "AvR", "AvL", "AvF", 
                  "AVr", "AVl", "AVf", 
                  "avR", "avL", "avF",
                  "aVr", "aVl", "aVf",
                  "Avr", "Avl", "Avf",
                  "avr", "avl", "avf",
                  "AVR", "AVL", "AVF",
                  "V1", "V2", "V3", "V4", "V5", "V6", 
                  "V1", "V2", "V3", "V4", "V5", "V6", 
                  "V1", "V2", "V3", "V4", "V5", "V6", 
                  "v1", "v2", "v3", "v4", "v5", "v6",
                  "v1", "v2", "v3", "v4", "v5", "v6",
                  "v1", "v2", "v3", "v4", "v5", "v6",
                  "C1", "C2", "C3", "C4", "C5", "C6", 
                  "c1", "c2", "c3", "c4", "c5", "c6"                 
                 ]

    # List of distractor characters
    distractor_chars = list("BDEGHJKMNOPQSTUWXYZ7890")

    # Directories
    fonts_dir = "fonts/" # Directory with fonts
    img_dir = "datasets/data_full/images/train/" # Directory with ECG images
    json_dir = "datasets/data_full/json/" # Directory with corresponding labels for lead name bounding boxes
    img_lead_dir = "datasets/data_lead/images/train" # Directory to save augmented images
    label_dir = "datasets/data_lead/labels/train" # Directory to save augmented labels

    # Make directories if they don't exist
    os.makedirs(img_lead_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Load all font file paths from the 'fonts' directory
    font_paths = load_fonts(fonts_dir)
    
    # Parameters for number of lead names and distractor characters to add
    max_extra_leads = 10
    max_distractors = 3

    all_images = glob.glob(os.path.join(img_dir, "*.png"))

    # Use 20,000 images
    sampled_images = random.sample(all_images, min(20000, len(all_images)))

    args_list = [
        (img_path, font_paths, lead_names, distractor_chars, max_extra_leads, max_distractors, label_dir, json_dir, img_lead_dir)
        for img_path in sampled_images
    ]

    # Use a multiprocessing pool to speed up processing across images
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, args_list), total=len(args_list), desc="Processing images"))

if __name__ == "__main__":
    main()


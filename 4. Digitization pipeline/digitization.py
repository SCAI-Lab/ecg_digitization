import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from scipy.interpolate import interp1d
import wfdb
from scipy import signal
from skimage import morphology, segmentation
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
from skimage.filters import threshold_multiotsu
import re

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,
)

def plot_image(img, title="Image Plot", size=(12, 12), show_axis=False):
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if not show_axis:
        plt.axis('off')
    plt.show()

def shadow_removal(img):
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 15)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def parse_layout_from_folder(folder_path):    
    base_name = folder_path.split('/')[-1]
    
    match = re.search(r'ecg_signals_(\d+)x(\d+)_(None|Rythm)_(Cabrera|Normal)', base_name)
    if match:
        rows = int(match.group(1))
        cols = int(match.group(2))
        calibration = (match.group(3) == 'Rythm')    
        cabrera_flag = (match.group(4) == 'Cabrera')   
        layout_key = (rows, cols, cabrera_flag)
        return layout_key, calibration
    else:
        return None, None

class ECGImage:
    def __init__(self, box_model, segmentation_model, lead_name_model, pulse_model, image_path, wfdb_path=""):
        self.load_image(image_path)
        self.wfdb_path = wfdb_path

        self.box_model = YOLO(box_model)
        self.segmentation_model = segmentation_model
        self.lead_name_model = YOLO(lead_name_model)
        self.pulse_model = YOLO(pulse_model)

        self.is_cabrera = None
        self.has_calibration_pulse = None
        
        self.standard_layouts = {
            # Standard layouts
            (12, 1, False): [['I'], ['II'], ['III'], ['aVR'], ['aVL'], ['aVF'], ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
            (6, 2, False): [['I', 'V1'], ['II', 'V2'], ['III', 'V3'], ['aVR', 'V4'], ['aVL', 'V5'], ['aVF', 'V6']],
            (4, 3, False): [['I', 'II', 'III'], ['aVR', 'aVL', 'aVF'], ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6']],
            (3, 4, False): [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']],
            
            # Cabrera layouts
            (12, 1, True): [['aVL'], ['I'], ['aVR'], ['II'], ['aVF'], ['III'], ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
            (6, 2, True): [['aVL', 'V1'], ['I', 'V2'], ['aVR', 'V3'], ['II', 'V4'], ['aVF', 'V5'], ['III', 'V6']],
            (4, 3, True): [['aVL', 'I', 'aVR'], ['II', 'aVF', 'III'], ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6']],
            (3, 4, True): [['aVL', 'II', 'V1', 'V4'], ['I', 'aVF', 'V2', 'V5'], ['aVR', 'III', 'V3', 'V6']]
        }

    def load_image(self, path, target_size=1700):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        resample_factor = target_size / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*resample_factor), int(img.shape[0]*resample_factor)), interpolation=cv2.INTER_CUBIC)

        self.image = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def preprocess_image(self):

        rem = shadow_removal(self.image)
        self.processed_image = cv2.GaussianBlur(rem, (3, 3), 0)

    def segment_leads(self):

        segmentations = []

        for shape in [4, 4.5, 5]:
            element_crops = MakeCropsDetectThem(
                image=cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR),
                model_path=self.segmentation_model,
                segment=True,
                show_crops=False,
                shape_x=int(self.processed_image.shape[0] // shape),
                shape_y=int(self.processed_image.shape[0] // shape),
                overlap_x=50,
                overlap_y=50,
                conf=0.8,
                iou=0.7,
                classes_list=[0],
            )
            
            segmentations.append(CombineDetections(element_crops, nms_threshold=0.5))
        
        self.lead_segmentation = segmentations
    
    def make_segmentation_mask(self):
        height, width = self.image.shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)
    
        for segmentation in self.lead_segmentation:
            polygons = segmentation.filtered_polygons
    
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                if pts.ndim == 2:
                    pts = [pts]
                cv2.fillPoly(combined_mask, pts, color=255)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        self.mask_image = img    
    
    def find_row_centers(self):
        var_image = self.mask_image
        image_height, image_width = self.mask_image.shape[:2] 
        
        proj = np.sum(var_image, 1)
        
        height = (image_width // 10) * 255
        distance = image_height // 30
        
        peaks, _ = find_peaks(proj, height=height, distance=distance)  
        
        row_centers, _ = find_peaks(proj, height=height, distance=int((np.mean(np.diff(peaks)) * (2/3))))
        
        self.row_centers = row_centers
        
        if len(peaks) > 0:
            first_peak = peaks[0]
            start_index = first_peak
            zero_gap = 0
            for i in range(first_peak - 1, -1, -1):
                if proj[i] == 0:
                    zero_gap += 1
                    if zero_gap > 2:
                        break
                else:
                    zero_gap = 0
                    start_index = i
            self.first_peak_start = start_index
        
            last_peak = peaks[-1]
            end_index = last_peak
            zero_gap = 0
            for i in range(last_peak + 1, image_height):
                if proj[i] == 0:
                    zero_gap += 1
                    if zero_gap > 2:
                        break
                else:
                    zero_gap = 0
                    end_index = i
            self.last_peak_end = end_index


    def get_roi(self):

        spacing = 2/3*np.mean(np.diff(self.row_centers))

        min_y = max(0, self.row_centers[0] - spacing)
        max_y = min(self.image.shape[0], self.row_centers[-1] + spacing)

        self.roi = (min_y, max_y)


    def extract_lead_boxes(self):
        results = self.box_model.predict(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR), conf=0.8, verbose=False)
        
        min_y, max_y = self.roi
        
        lead_boxes = []
    
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
    
                if y1 >= min_y and y2 <= max_y:
                    lead_boxes.append([x1, y1, x2, y2])
        
        self.lead_bboxes = lead_boxes

    def extract_lead_name_boxes(self):
        results = self.lead_name_model.predict(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR), conf=0.8, verbose=False)
        
        min_y, max_y = self.roi

        name_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
                bbox = box.xyxy.cpu().numpy()[0].tolist()

                if y1 >= min_y and y2 <= max_y:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    cls_name = self.lead_name_model.names[cls_id]
    
                    name_boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': cls_name
                    })
    
        self.lead_name_bboxes = name_boxes


    def extract_reference_pulses(self):
        results = self.pulse_model.predict(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR), conf=0.8, verbose=False)
        
        pulse_boxes = []
        for r in results:
            for box in r.boxes:
                coord = box.xyxy.cpu().numpy()[0].tolist()
                pulse_boxes.append({
                    'image': self.image[int(coord[1])-5:int(coord[3])+5, int(coord[0])-5:int(coord[2])+5],
                    'bbox': coord
                })
        
        self.reference_pulses = pulse_boxes

    
    def visualize_boxes(self, task='Lead name', show_axis=False):
        img_copy = self.image.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        
        if task == 'Lead name':
            if self.lead_name_bboxes is None:
                print("Lead name boxes not extracted.")
                return
            
            for box in self.lead_name_bboxes:
                x1, y1, x2, y2 = map(int, box['bbox'])
                label = box['class_name']
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_copy, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
        elif task == 'Lead box':
            if self.lead_bboxes is None:
                print("Lead boxes not extracted.")
                return
            
            for bbox in self.lead_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        elif task == 'Reference pulse':
            if self.reference_pulses is None:
                print("Reference pulses not extracted.")
                return
            
            for bbox in self.reference_pulses:
                x1, y1, x2, y2 = map(int, bbox['bbox'])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        else:
            print(f"Unknown task: {task}")
            return
        
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(img_rgb)
        if not show_axis:
            plt.axis('off')
        plt.show()

    def visualize_segmentation(self, show_boxes=False, show_axis=False, fill_mask=True, thickness=1):
        all_confidences = []
        all_boxes = []
        all_polygons = []
        all_classes_ids = []
        all_classes_names = []
    
        for segmentation in self.lead_segmentation:
            all_confidences.extend(segmentation.filtered_confidences)
            all_boxes.extend(segmentation.filtered_boxes)
            all_polygons.extend(segmentation.filtered_polygons)
            all_classes_ids.extend(segmentation.filtered_classes_id)
            all_classes_names.extend(segmentation.filtered_classes_names)
    
        visualize_results(
            img=cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB),
            confidences=all_confidences,
            boxes=all_boxes,
            polygons=all_polygons,
            classes_ids=all_classes_ids,
            classes_names=all_classes_names,
            segment=True,
            thickness=thickness,
            fill_mask=fill_mask,
            show_boxes=show_boxes,
            show_class=False,
            axis_off=(not show_axis)
        )

    def get_reference_scale(self):
        voltages = []
        times = []
        times_line = []
        dist = []
       
        def line_length(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for i, pulse in enumerate(self.reference_pulses):
            img = pulse['image']
            h, w = img.shape
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
            
            blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
            
            thresholds = threshold_multiotsu(blurred, classes=2) 
            regions = np.digitize(blurred, bins=thresholds)
            binary = (regions == 0).astype(np.uint8) * 255
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            line_mask = cv2.bitwise_or(horizontal_lines_img, vertical_lines_img)
        
            horizontal_lines = cv2.HoughLinesP(
                horizontal_lines_img,
                rho=1,
                theta=np.pi / 180,
                threshold= w // 4,
                minLineLength= w // 2,  
                maxLineGap=1
            )
            
            vertical_lines = cv2.HoughLinesP(
                vertical_lines_img,
                rho=1,
                theta=np.pi / 180,
                threshold= h // 4,
                minLineLength= h // 2, 
                maxLineGap=1
            )
            
            combined_lines = []
            
            if horizontal_lines is not None:
                combined_lines.extend(horizontal_lines)  
            
            if vertical_lines is not None:
                combined_lines.extend(vertical_lines)  
            
            horizontal_lines = []
            vertical_lines = []
            
            if combined_lines is not None:
                for line in combined_lines:
                    x1, y1, x2, y2 = line[0]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
            
                    angle = np.arctan2(dy, dx) * 180 / np.pi
            
                    if angle < 10 or angle > 170:
                        horizontal_lines.append((x1, y1, x2, y2))
                    elif 80 < angle < 100:
                        vertical_lines.append((x1, y1, x2, y2))
           
            longest_verticals = sorted(vertical_lines, key=lambda l: line_length(*l), reverse=True)[:2]
            longest_horizontal = sorted(horizontal_lines, key=lambda l: line_length(*l), reverse=True)[:1]
        
            output = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
            
            x1 = (longest_verticals[0][0] + longest_verticals[0][2]) / 2
            x2 = (longest_verticals[1][0] + longest_verticals[1][2]) / 2
            vertical_spacing = abs(x2 - x1)
            
            voltages.append(sorted([line_length(*l) for l in vertical_lines], reverse=True)[:2])
        
            if vertical_spacing > 5:
                times.append(vertical_spacing)
            times_line.append(sorted([line_length(*l) for l in horizontal_lines], reverse=True)[:1])
        
        
            output_ = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        
            lsd = cv2.createLineSegmentDetector(refine=2)
            lines, _, _, _ = lsd.detect(vertical_lines_img)
        
            min_length = h / 3
            angle_tolerance_deg = 5
            
            angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)
            
            filtered_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.hypot(dx, dy)
                    
                    if length >= min_length:
                        angle = np.arctan2(dy, dx)  
            
                        if np.abs(np.abs(angle) - np.pi/2) <= angle_tolerance_rad:
                            filtered_lines.append([[x1, y1, x2, y2]])
        
            if len(filtered_lines) >= 4:
        
                def line_length_(line):
                    x1, y1, x2, y2 = line[0]
                    return np.hypot(x2 - x1, y2 - y1)
            
                filtered_lines.sort(key=line_length_, reverse=True)
                filtered_lines_np = filtered_lines[:4]
                
                def x_center(line):
                    x1, _, x2, _ = line[0]
                    return (x1 + x2) / 2
            
                filtered_lines_np.sort(key=x_center)
            
                pairs = [(filtered_lines_np[0], filtered_lines_np[1]), (filtered_lines_np[2], filtered_lines_np[3])]
            
                midpoints = []
                for l1, l2 in pairs:
                    x1 = (l1[0][0] + l1[0][2]) / 2
                    x2 = (l2[0][0] + l2[0][2]) / 2
                    x_mid = (x1 + x2) / 2
                    midpoints.append(x_mid)
        
                dist.append(midpoints[1]-midpoints[0])
                filtered_lines_np = np.array(filtered_lines_np, dtype=np.float32)
                        
        self.volt_per_pixel = 1 / np.mean(voltages)
        self.time_per_pixel = 0.2 / np.mean(dist)
        

    def make_bounding_box_features(self, box, axis):
        if axis == 'y':
            axis_min, axis_max = box[1], box[3]
        else:
            axis_min, axis_max = box[0], box[2]
        
        axis_center = (axis_min + axis_max) / 2

        return [axis_min, axis_center, axis_max]
        

    def bounding_boxes_kmeans(self, bounding_boxes, axis='y', k_min=1, k_max=13, return_model=True):
        if axis not in ('x', 'y'):
            raise ValueError("Axis must be 'x' or 'y'")
    
        if len(bounding_boxes) < k_min:
            raise ValueError("Not enough bounding boxes to cluster")
    
        features = []
        for box in bounding_boxes:
            features.append(self.make_bounding_box_features(box, axis))
            
        features = np.array(features)
    
        best_score = -1
        best_k = k_min
        best_labels = None
        best_centers = None
        best_model = None
    
        for k in range(k_min, min(k_max + 1, len(bounding_boxes))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(features)
    
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_centers = kmeans.cluster_centers_
                best_model = kmeans 
    
        cluster_avgs = best_centers.mean(axis=1)
        sorted_indices = np.argsort(cluster_avgs)
        label_map = {old: new for new, old in enumerate(sorted_indices)}
        sorted_labels = np.array([label_map[label] for label in best_labels])
        sorted_centers = cluster_avgs[sorted_indices]

        if return_model:
            return sorted_labels, best_k, sorted_centers, label_map, best_model
        else:
            return sorted_labels, best_k, sorted_centers, label_map

    def check_cabrera(self, num_rows, num_cols):

        if num_rows in [13, 12, 7, 6]:
            av_leads = [box for box in self.lead_name_bboxes if box['class_name'] in {'aVR', 'aVL', 'aVF'}]
            v_leads = [box for box in self.lead_name_bboxes if box['class_name'] in {'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}]

            if len(av_leads) == 0 or len(v_leads) == 0:
                return False

            y_coords_temp = []
            for lead in v_leads:
                _, y1, _, y2 = lead['bbox']
                cy = (y1 + y2) / 2
                y_coords_temp.append(cy)

            y_coords = []
            for lead in av_leads:
                _, y1, _, y2 = lead['bbox']
                cy = (y1 + y2) / 2
                y_coords.append(cy)

            y_coords_temp.sort()
            y_coords.sort()
            
            threshold = 30
            
            diff_temp = np.diff(y_coords_temp)
            diff_temp = diff_temp[np.abs(diff_temp) > threshold]
            
            diff_y = np.diff(y_coords)
            diff_y = diff_y[np.abs(diff_y) > threshold]

            med_dist = np.min(diff_temp)
            dist_y = np.min(diff_y)

            if abs(med_dist - dist_y) > (0.25*med_dist):
                self.is_cabrera = True
                return True
            else:
                self.is_cabrera = False
                return False
            
        elif (num_rows == 4 and num_cols == 3) or num_rows == 5:
            av_leads = [box for box in self.lead_name_bboxes if box['class_name'] in {'aVR', 'aVL', 'aVF'}]

            if len(av_leads) == 0:
                return False

            y_coords = []
            for lead in av_leads:
                _, y1, _, y2 = lead['bbox']
                cy = (y1 + y2) / 2
                y_coords.append(cy)
            
            std_y = np.std(y_coords)
            
            if std_y > 25:
                self.is_cabrera = True
                return True
            else:
                self.is_cabrera = False
                return False

        elif (num_rows == 4 and num_cols == 4) or num_rows == 3:
            av_leads = [box for box in self.lead_name_bboxes if box['class_name'] in {'aVR', 'aVL', 'aVF'}]

            if len(av_leads) == 0:
                return False

            x_coords = []
            for lead in av_leads:
                x1, _, x2, _ = lead['bbox']
                cx = (x1 + x2) / 2
                x_coords.append(cx)
            
            std_x = np.std(x_coords)
            
            if std_x > 25:
                self.is_cabrera = True
                return True
            else:
                self.is_cabrera = False
                return False
                        
    def get_layout(self, num_rows):

        if num_rows == 13:
            num_cols = 1
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(12, 1, True)]
            else:
                self.layout = self.standard_layouts[(12, 1, False)]
            
        elif num_rows == 12:
            num_cols = 1
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(12, 1, True)]
            else:
                self.layout = self.standard_layouts[(12, 1, False)]
            
        elif num_rows == 7:
            num_cols = 2
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(6, 2, True)]
            else:
                self.layout = self.standard_layouts[(6, 2, False)]
            
        elif num_rows == 6:
            num_cols = 2
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(6, 2, True)]
            else:
                self.layout = self.standard_layouts[(6, 2, False)]
            
        elif num_rows == 5:
            num_cols = 3
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(4, 3, True)]
            else:
                self.layout = self.standard_layouts[(4, 3, False)]
            
        elif num_rows == 4:

            score_normal = 0
            score_cabrera = 0
            v_leads1 = [box for box in self.lead_name_bboxes if box['class_name'] in {'V1', 'V2', 'V3'}]
            v_leads2 = [box for box in self.lead_name_bboxes if box['class_name'] in {'V4', 'V5', 'V6'}]
            
            def centers(boxes, axis):
                if axis == 'x':
                    return [(box['bbox'][0] + box['bbox'][2]) / 2 for box in boxes]
                elif axis == 'y':
                    return [(box['bbox'][1] + box['bbox'][3]) / 2 for box in boxes]
        
            x_std_leads1 = np.std(centers(v_leads1, 'x'))
            y_std_leads1 = np.std(centers(v_leads1, 'y'))
            x_std_leads2 = np.std(centers(v_leads2, 'x'))
            y_std_leads2 = np.std(centers(v_leads2, 'y'))
        
            if x_std_leads1 < y_std_leads1 and x_std_leads2 < y_std_leads2:
                num_cols = 4
                self.has_calibration_pulse = True

            elif y_std_leads1 < x_std_leads1 and y_std_leads2 < x_std_leads2:
                num_cols = 3
                self.has_calibration_pulse = False

            else:
                num_cols = 4
                self.has_calibration_pulse = True

            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                if num_cols == 4:
                    self.layout = self.standard_layouts[(3, 4, True)]
                else:
                    self.layout = self.standard_layouts[(4, 3, True)]
            else:
                if num_cols == 4:
                    self.layout = self.standard_layouts[(3, 4, False)]
                else:
                    self.layout = self.standard_layouts[(4, 3, False)]
            
        elif num_rows == 3:
            num_cols = 4
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            if cabrera:
                self.layout = self.standard_layouts[(3, 4, True)]
            else:
                self.layout = self.standard_layouts[(3, 4, False)]

        return num_cols

    def make_grid(self, padding=0):

        image_height, image_width = self.image.shape[:2] 

        num_rows = len(self.row_centers)
        num_cols = self.get_layout(num_rows)
        
        row_masks = {i: np.zeros((image_height, image_width), dtype=np.uint8) for i in range(num_rows)}
        row_polygons = {i: [] for i in range(num_rows)}
        row_limits = {i: [] for i in range(num_rows)}
        
        for segmentation in self.lead_segmentation:
            for idx, (box, polygon) in enumerate(zip(segmentation.filtered_boxes,
                                                     segmentation.filtered_polygons)):
                
                poly_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                temp_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [poly_np], color=1)
        
                temp_proj = np.sum(temp_mask, axis=1)
                y_indices = np.arange(temp_proj.shape[0])
                total_weight_y = np.sum(temp_proj)
        
                if total_weight_y == 0:
                    continue
        
                centroid_y = int(np.sum(y_indices * temp_proj) / total_weight_y)
        
                y_vals = [pt[1] for pt in polygon]
                min_y = min(y_vals)
                max_y = max(y_vals)
        
                if max_y < self.first_peak_start or min_y > self.last_peak_end:
                    continue
        
                diffs = np.abs(self.row_centers - centroid_y)
                closest_idx = np.argmin(diffs)
        
                if self.has_calibration_pulse and closest_idx == num_rows - 1:
                    continue
        
                row_polygons[closest_idx].append(polygon)
                cv2.fillPoly(row_masks[closest_idx], [poly_np], color=1)
        
                if not row_limits[closest_idx]:
                    row_limits[closest_idx] = [min_y, max_y]
                else:
                    current_min, current_max = row_limits[closest_idx]
                    row_limits[closest_idx][0] = min(current_min, min_y)
                    row_limits[closest_idx][1] = max(current_max, max_y)

        cropped_row_masks = {}
        
        for i in range(num_rows):
            if self.has_calibration_pulse and i == num_rows - 1:
                continue 
        
            mask = row_masks[i]
            limits = row_limits[i]
        
            if not limits:
                continue
        
            min_y, max_y = limits
            cropped_mask = mask[min_y:max_y+1, :]
        
            cropped_row_masks[i] = cropped_mask


        if self.has_calibration_pulse:
            lead_boxes = []
        
            for box in self.lead_bboxes:
                x1, y1, x2, y2 = map(int, box)
                y_center = (y1 + y2) // 2
        
                diffs = np.abs(self.row_centers - y_center)
                closest_idx = int(np.argmin(diffs))
        
                if closest_idx != (num_rows - 1):
                    lead_boxes.append(box)
        else:
            lead_boxes = self.lead_bboxes

        self.row = cropped_row_masks
        
        if num_cols != 1:
            labels_cols, _, _, _ = self.bounding_boxes_kmeans(lead_boxes, axis='x', k_min=num_cols, k_max=num_cols, return_model=False)
        else:
            labels_cols = np.array([0 for _ in range(len(lead_boxes))])
            
        min_x_per_col = []
        max_x_per_col = []

        boxes_arr = np.array(lead_boxes)

        for col_label in range(num_cols):

            col_boxes = boxes_arr[labels_cols == col_label]
            
            min_x = col_boxes[:, 0].min()
            max_x = col_boxes[:, 2].max()
            
            min_x_per_col.append(min_x)
            max_x_per_col.append(max_x)

        mask_grid = []
        relative_baselines = []

        for row_idx, row_slice in cropped_row_masks.items():
            
            row_cells = []
            for col_idx in range(num_cols):
                x_min = min_x_per_col[col_idx]
                x_max = max_x_per_col[col_idx]
            
                slice_x_min = max(0, int(x_min))
                slice_x_max = min(image_width, int(x_max))

                row_column_slice = row_slice[:, slice_x_min:slice_x_max]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                row_column_slice = cv2.morphologyEx(row_column_slice, cv2.MORPH_OPEN, kernel)
                
                row_cells.append({
                    'lead': self.layout[row_idx][col_idx],
                    'signal': row_column_slice
                })
                
            relative_baselines.append(self.row_centers[row_idx] - row_limits[row_idx][0])            
            mask_grid.append(row_cells)

        self.grid = mask_grid
        self.baseline = relative_baselines
    
    def visualize_grid(self, figsize=(15, 10)):
        if not hasattr(self, 'grid') or not self.grid:
            raise ValueError("Grid not generated. Call make_grid() first.")
    
        grid = self.grid
        num_rows = len(grid)
        num_cols = len(grid[0]) if isinstance(grid[0], list) else 1
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
        if num_cols == 1:
            axes = np.atleast_2d(axes).T
        elif num_rows == 1:
            axes = np.atleast_2d(axes)
    
        for row_idx in range(num_rows):
            row = grid[row_idx] if isinstance(grid[row_idx], list) else [grid[row_idx]]
            for col_idx, cell in enumerate(row):
                ax = axes[row_idx][col_idx]
                ax.imshow(cell['signal'], cmap='gray', aspect='auto')
                ax.set_title(cell['lead'], fontsize=10)
                ax.axis('off')
    
        plt.tight_layout()
        plt.show()

    def binarize_signal(self, img, window_length=11, polyorder=2):
    
        height, width = img.shape
        x_coords = []
        initial_signal = []
    
        for col in range(width):
            column = img[:, col]
            if np.sum(column) == 0:
                continue
                
            y_indices = np.arange(height)
            weights = column.astype(float)
            centroid = np.average(y_indices, weights=weights)
            initial_signal.append(centroid)
            x_coords.append(col)

        x_coords = np.array(x_coords)
        initial_signal = np.array(initial_signal)
    
        second_deriv = np.gradient(np.gradient(savgol_filter(initial_signal, window_length, polyorder)))
        second_deriv = savgol_filter(second_deriv, 11, 3)
        
        x_out, final_signal = [], []
        for idx, col in enumerate(x_coords):
            column = img[:, col]
            nz_idx = np.where(column > 0)[0]            
            if nz_idx.size == 0:
                continue
    
            if second_deriv[idx] > 0.5:                    
                sel_idx = nz_idx[:5]                    
            elif second_deriv[idx] < -0.5:                                       
                sel_idx = nz_idx[-5:]    
            else: 
                sel_idx = nz_idx
    
            weights  = column[sel_idx].astype(float)
            centroid = np.average(sel_idx, weights=weights)
    
            x_out.append(col)
            final_signal.append(centroid)
    
        return np.array(x_out), np.array(final_signal)

    def fill_gaps(self, x_coords, y_coords, method='linear'):
        x_full = np.arange(x_coords[0], x_coords[-1] + 1)
        interpolator = interp1d(x_coords, y_coords, kind=method, fill_value="extrapolate")
        y_interp = interpolator(x_full)
        return x_full, y_interp

    def smooth_signal(self, signal, window_length=7, polyorder=4):
        smoothed = savgol_filter(signal, window_length, polyorder)
        return smoothed
            
    def extract_signals(self):
        signal_grid = []
        sample_rate = 500
        for row_idx, row in enumerate(self.grid):
            baseline_y = self.baseline[row_idx]   
            row_signals = []
            
            for cell in row:
                x_coords, signal = self.binarize_signal(cell['signal'])

                x_coords = x_coords[5:-5]
                signal = signal[5:-5]
                
                x_coords, signal = self.fill_gaps(x_coords, signal, method='linear')
                signal = self.smooth_signal(signal)
                
                signal_volts = (baseline_y - signal) * self.volt_per_pixel
                x_seconds = x_coords * self.time_per_pixel
    
                duration = x_seconds[-1] - x_seconds[0]
                num_samples = int(duration * sample_rate) + 1
                resampled_time = np.linspace(x_seconds[0], x_seconds[-1], num_samples)
    
                interpolator = interp1d(x_seconds, signal_volts, kind='linear', fill_value="extrapolate")
                resampled_signal = interpolator(resampled_time)
    
                row_signals.append({
                    'lead': cell['lead'],
                    'time': resampled_time,
                    'signal': resampled_signal
                })
    
            signal_grid.append(row_signals)
    
        self.signal_grid = signal_grid
        
    def sliding_metrics(self, signal_a, signal_b, return_aligned_signals=False):
        len_a, len_b = len(signal_a), len(signal_b)
        if len_a > len_b:
            long_signal, short_signal = signal_a, signal_b
        else:
            long_signal, short_signal = signal_b, signal_a
    
        len_long, len_short = len(long_signal), len(short_signal)
        
        max_corr, best_corr_offset = -1, 0
    
        for i in range(len_long - len_short + 1):
            window = long_signal[i:i+len_short]
            corr, _ = pearsonr(window, short_signal)
            if corr > max_corr:
                max_corr = corr
                best_corr_offset = i
    
        best_window = long_signal[best_corr_offset:best_corr_offset + len_short]
    
        rmse = mean_squared_error(best_window, short_signal)
        signal_power = np.mean(np.square(short_signal))
        noise_power = np.mean(np.square(best_window - short_signal))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    
        if return_aligned_signals:
            return max_corr, rmse, snr, short_signal, best_window
        return max_corr, rmse, snr
    
    def calculate_metrics_ptb(self, plot_signals=True, per_lead_scores=None):
        record = wfdb.rdrecord(self.wfdb_path)
    
        avg_pearson, avg_rmse, avg_snr = [], [], []
    
        for row in self.signal_grid:
            for cell in row:
                if 'lead' not in cell or 'signal' not in cell:
                    continue
    
                try:
                    lead_index = record.sig_name.index(cell['lead'])
                except ValueError:
                    continue
    
                wfdb_signal = record.p_signal[:, lead_index]
                wfdb_signal = wfdb_signal[~np.isnan(wfdb_signal)]
                voltage_signal = np.array(cell['signal'])
    
                pearson_val, rmse, snr, sig1, sig2 = self.sliding_metrics(
                    voltage_signal, wfdb_signal, return_aligned_signals=True
                )
    
                try:
                    pearson, pval = pearsonr(sig1, sig2)
                except:
                    pearson, pval = np.nan, np.nan
    
                cell['pearson'] = pearson
                cell['rmse'] = rmse
                cell['snr'] = snr
                cell['pval'] = pval
    
                avg_pearson.append(pearson)
                avg_rmse.append(rmse)
                avg_snr.append(snr)
    
                if per_lead_scores is not None and pearson > 0.60:
                    lead = cell['lead']
                    if lead not in per_lead_scores:
                        per_lead_scores[lead] = {'pearson': [], 'rmse': [], 'snr': [], 'pval': []}
                    per_lead_scores[lead]['pearson'].append(pearson)
                    per_lead_scores[lead]['rmse'].append(rmse)
                    per_lead_scores[lead]['snr'].append(snr)
                    per_lead_scores[lead]['pval'].append(pval)
    
        self.average_pearson = np.mean(avg_pearson)
        self.average_rmse = np.mean(avg_rmse)
        self.average_snr = np.mean(avg_snr)
    
    def plot_signals(self, title='', plot_wfdb=False):    
        for row in self.signal_grid:
            for cell in row:
                
                voltage_signal = cell['signal']

                if plot_wfdb:
                    lead_index = record.sig_name.index(cell['lead'])
                    wfdb_signal = record.p_signal[:, lead_index]
                    wfdb_signal = [x for x in wfdb_signal if not math.isnan(x)]
                    _, _, _, sig1, sig2 = self.sliding_metrics(voltage_signal_std, wfdb_signal_std, return_aligned_signals=True)
    
                plt.figure(figsize=(10, 4))
                if not plot_wfdb:
                    plt.plot(voltage_signal, linewidth=1.5)
                if plot_wfdb:
                    plt.plot(sig1, label='Extracted Signal', linewidth=1.5)
                    plt.plot(sig2, label='Ground Truth', linewidth=1.5)
                plt.title(title)
                plt.legend()
                plt.xlabel("Time (ms)")
                plt.ylabel("Voltage (mV)")
                plt.tight_layout()
                plt.show()
                    
    def save_signals_as_wfdb(self, record_name, directory='.'):
        signals = []
        lead_names = []
    
        max_length = 0
    
        for row in self.signal_grid:
            for cell in row:
                if 'signal' in cell and 'lead' in cell:
                    sig = np.array(cell['signal'])
                    signals.append(sig)
                    lead_names.append(cell['lead'])
                    max_length = max(max_length, len(sig))
    
        padded_signals = []
        for sig in signals:
            padded = np.pad(sig, (0, max_length - len(sig)), mode='constant', constant_values=np.nan)
            padded_signals.append(padded)
    
        signal_array = np.array(padded_signals).T
    
        signal_array = np.nan_to_num(signal_array, nan=0.0)
    
        fs = 500  
        units = ['mV'] * len(lead_names)
        fmt = ['16'] * len(lead_names)  
    
        wfdb.wrsamp(
            record_name=record_name,
            fs=fs,
            units=units,
            sig_name=lead_names,
            p_signal=signal_array,
            fmt=fmt,
            write_dir=directory
        )

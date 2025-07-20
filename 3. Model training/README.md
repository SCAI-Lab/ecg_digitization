## Model training

Each model can be trained using the associated file:

1. yolo_train_full.py: train the YOLO model on the full image dataset
2. yolo_train_patch.py: train the YOLO model on the patched image dataset
3. yolo_train_lead.py: train the YOLO model on the lead name image dataset
4. yolo_train_pulse.py: train the YOLO model on the reference pulse image dataset

Results are saved in the [runs](/.runs/) folder

## Installation

### Setup with Conda Environment

```bash
conda env create -f environment.yml
conda activate yolo11

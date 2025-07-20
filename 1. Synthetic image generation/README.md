## Synthethic image generation

In order to generate the dataset from the PTB-XL dataset (or any other dataset with the same format), WFDB formatted files (pairs of .hea and .dat files) need to be placed in the all_signals folder. The images can then be generated using the generate_images.sh bash file. 

To modify whether the images are generated in Cabrera format and if a rhythm strip is present, change the following lines in the `ecg_plot.py` file:

- Line 235: use_cabrera = True/False 
- Line 128: #full_mode = 'None' : either comment or uncomment it (if commented, rhythm strip will be added)

To modify the layout, change the *--num_columns* argument in the `generate_images.sh` file to either 1, 2, 3 or 4

## Installation

### Setup with Conda Environment

```bash
conda env create -f environment.yml
conda activate ecg

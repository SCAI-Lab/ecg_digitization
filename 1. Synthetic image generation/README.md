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
```
## Code adapted from [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit/)

Kshama Kodthalu Shivashankara, Deepanshi, Afagh Mehri Shervedani, Matthew A. Reyna, Gari D. Clifford, Reza Sameni (2024). ECG-image-kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. In Physiological Measurement. IOP Publishing. doi: 10.1088/1361-6579/ad4954

ECG-Image-Kit: A Toolkit for Synthesis, Analysis, and Digitization of Electrocardiogram Images, (2024). URL: https://github.com/alphanumericslab/ecg-image-kit

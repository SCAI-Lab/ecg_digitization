## Digitization pipeline

The digitization pipeline code is found in the `digitization.py` file, and the code to run it is found in the `digitize_image.ipynb` notebook. The notebook contains three different run options:

1. Digitize a single image, visualie the results and store the time series in WFDB format
2. Run the layout detection test to get the accuracy 
3. Run the digitization performance test to get the Pearson Correlation Coefficient, RMSE and SNR

The [test_set](/.test_set/) folder contains the images to test the digitization performance, and the other folders contain the test images for each layout

## Installation

### Setup with Conda Environment

```bash
conda env create -f environment.yml
conda activate infer

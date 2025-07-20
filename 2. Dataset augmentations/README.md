## Dataset augmentations

Three python files are used to make the different datasets:

1. make_leadname.py: used to generate the lead name dataset
2. make_patches.py: used to generate the patched version of the synthetic dataset
3. make_pulse.pyused to generate the reference pulse dataset (requires images without other reference pulses present in the [images_without_pulse](/.images_without_pulse/) folder, and drawn reference pulses for augmentation in the [pulses](/pulses/) folder)

## Installation

### Setup with Conda Environment

```bash
conda env create -f environment.yml
conda activate pul

## ECG Digitization
***Development of an ECG digitization pipeline using deep learning***

Author: Cyrus Achtari (cachtari@student.ethz.ch)

The code consists of 4 folders, each containing an individual step in developing the digitization pipeline. The order must be followed as the output of previous steps is required at each stage. 

1. Synthetic image generation: generating the synthetic ECG dataset from the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset
2. Dataset augmentations: augmenting the previously made dataset for each model
3. Model Training: python files to run the model training
4. Digitization pipeline: notebooks to digitize single images, or run validation metrics on the test sets 

README files are found within each folder with instructions for running each process and the environment used as different environemnts had to be setup due to compatibility issues

## intermediate_steps.ipynb

The intermediate steps between each folder are found in the intermediate_steps.ipynb notebook, which is used to move around the files and visualize the results at each step to ensure proper functioning of the code. 

The libraries needed for runing the scripts are:
  * os,
  * tqdm,
  * shutil,
  * random,
  * json,
  * cv2,
  * numpy,
  * matplotlib,
  * PIL,

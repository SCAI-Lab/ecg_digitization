## ECG Digitization

Author: Cyrus Achtari (cachtari@student.ethz.ch)

The code consists of 4 folders, each containing an individual step in developing the digitization pipeline. The intermediate steps between each folder are found in the intermediate_steps.ipynb notebook, which is used to move around the files and visualize the results at each step to ensure proper functioning of the code. 

The first step is to generate the various datasets used to train the three YOLOv11 models: a segmentation model to obtain the signal segmentation masks, and two object detection models to detect the lead names and the reference pulses. In order to generate the synthetic datasets, a certain order must be followed as some datasets require augmenting previously created ones. These steps are conducted in 1. Synthetic image generation and 2. Dataset augmentations

After having generated the datasets, the three models are trained using the 3. Model Training folder. 

Finally, the full digitization pipeline is found in 4. Digitization pipeline, along with notebooks to digitize single images, or run validation metrics on the test set. 

READ_ME files are found within each folder with instructions for running each process and the environment used as different environemnts had to be setup due to compatibility issues.

## Digitize_image.ipynb



The libraries needed for runing the scripts are:

  * matplotlib,
  * numpy,
  * pandas,
  * random,
  * scipy,
  * datatime,
  * sys,
  * os,
  * collections,

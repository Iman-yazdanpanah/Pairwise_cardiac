# Pairwise Segmentation
This code is an implementation of [pairwise semantic segmentation via conjugate fully convolutional network](https://link.springer.xilesou.top/chapter/10.1007/978-3-030-32226-7_18) and its journal version, which is implemented based on [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception/tree/previous) and experiments with the [Cardiac Catheterization X-Ray PNG 512x512](https://www.kaggle.com/datasets/c7934597/cardiac-catheterization) dataset.

# Get Started
## 1. Data preprocessing
Put the train folder in the root directory.  
## 2. Training and validation
To tain run the main_cfcn.py and change the neccessary arguments.

Once the model is trained, validate it by changing the only-val argument.

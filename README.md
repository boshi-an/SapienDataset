# Dataset for End-to-end Affordance Learning

The dataset used in End-to-end Affordance Learning is based on Partnet Mobility datset.

We selected cabinet, chair and drawers from the original dataset for different tasks in our experiment.

We processed each object in the following manner:

- For objects with multiple joints, create a new object for each joint, with all other joints fixed.
- Sample point clouds for each part.
- Compute bounding box and other info for each part.

## Processed Dataset

You can download the processed dataset from [GoogleDrive](https://drive.google.com/file/d/1PfMZ9KzLn2Z8mbGq9HfW58u6BnuDGT3W/view?usp=share_link) or manually build the dataset from original sapien Partnet Mobility dataset.

## Manually Process Dataset

Since our dataset was based on Sapien Partnet Mobility, you need to first download the original dataset from [Sapien](https://sapien.ucsd.edu/), and put the dataset in 'asset' directory inside the root of your clone of this repo.

To reproduce the data preparation process, run `create_xxx.py`.

You may need to modify the 'root' variable to let the program access the downloaded dataset.

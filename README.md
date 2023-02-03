# Dataset for End-to-end Affordance Learning

The dataset used in End-to-end Affordance Learning is based on Partnet Mobility datset.

We selected cabinet, chair and drawers from the original dataset for different tasks in our experiment.

We processed each object in the following manner:

- For objects with multiple joints, create a new object for each joint, with all other joints fixed.
- Sample point clouds for each part.
- Compute bounding box and other info for each part.

To reproduce the data preparation process, run `create_xxx.py`.
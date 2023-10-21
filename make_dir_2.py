import os
import random
import shutil

# Define the source directory
leafs_dir = "Leafs"

# Define the destination directory
dataset_dir = "Leafs_Structured"


# Loop through each leaf type folder
for leaf_type in os.listdir(leafs_dir):
    leaf_type_dir = os.path.join(leafs_dir, leaf_type)  # leadf/type

    for subfolder in ["healthy", "diseased"]:
        subfolder_path = os.path.join(leaf_type_dir, subfolder)  # leadf/type/sick
        image_files = os.listdir(subfolder_path)

        s = f"{dataset_dir}/{leaf_type+'_'+subfolder}"

        if not os.path.exists(s):
            os.makedirs(s)

        for file in image_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(s, file))


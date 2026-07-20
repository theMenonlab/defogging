import kagglehub

import sys


print("Getting Mapillary Vistas Dataset from Kaggle")
try:
    print(f"Downloading to {sys.argv[1]}") 
    print("This might take a while.")
    # Download latest version
    # not the best idea to not clean the argument but this is just a one
    # time use script to get the mapillary vistas dataset
    path = kagglehub.dataset_download("kaggleprollc/mapillary-vistas-image-data-collection", output_dir = sys.argv[1])

    print("Path to dataset files:", path)
# likely means nothign was passed to the program
except IndexError:
    print("ERROR: Pass the target directory to the program!!")
    print("Example: \npython3 getDataset.py <target directory>") 

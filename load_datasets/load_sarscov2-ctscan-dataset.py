import os

# Import and organizing sarscov2-ctscan-dataset
os.system("kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset")
os.system("unzip sarscov2-ctscan-dataset.zip")
os.system("mkdir sarscov2-ctscan-dataset")
os.system("mv COVID sarscov2-ctscan-dataset")
os.system("mv non-COVID sarscov2-ctscan-dataset")
os.system("rm -rf sarscov2-ctscan-dataset.zip")

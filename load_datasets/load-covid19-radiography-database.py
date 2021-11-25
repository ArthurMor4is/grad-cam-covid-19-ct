import os

# Import and organizing sarscov2-ctscan-dataset
os.system("kaggle datasets download -d tawsifurrahman/covid19-radiography-database")
os.system("unzip covid19-radiography-database.zip")
os.system("rm -rf covid19-radiography-database.zip")

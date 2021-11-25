## Carregar dataset
import os

# os.system("pip install torchxrayvision")

import torchxrayvision as xrv
import matplotlib.pyplot as plt
import pandas as pd


# os.system("git clone https://github.com/ieee8023/covid-chestxray-dataset")
d = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset/images/",csvpath="covid-chestxray-dataset/metadata.csv")

## Selecionando imagens de pascientes com covid
disease_label = 'COVID-19'
os.system("mkdir covid-chestxray-dataset/images/" + disease_label)
selected_images = []
for image in d:
    print(pd.Series(dict(zip(d.pathologies,image["lab"]))))
    if image['lab'][d.pathologies.index(disease_label)] == 1:
        selected_images.append(image)
        image_index = image['idx']
        file_name = d.csv.iloc[image_index]['filename']
        os.system("cp covid-chestxray-dataset/images/" + file_name + " covid-chestxray-dataset/images/" + disease_label)


## Existe um pasciente com mais de uma imagem ?

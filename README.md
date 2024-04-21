# Finding Mining Sites

This is a fundation model based solution for Solafune [Finding Mining Sites](https://solafune.com/competitions/58406cd6-c3bb-4f7a-85c7-c5a1ad67ca03?menu=about&tab=overview). The pretrained model is based on the [Prithvi-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M). Prithvi-100M is a temporal vision transformer model trained on US Harmonised Landsat Sentinel 2 (HLS) data, making it suitable for Sentinel 2 downstream tasks.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation
Download and unzip the training data from the competition page.

Rename the ```answer.csv``` to ```metadata.csv``` and put it in the training data folder.

Due to my git LFS quota, please download the pretrained model weight (Prithvi_100M.pt) from [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/tree/main) and specify the path of Prithvi_100M.pt in ```setting.json```.

## Log in to wandb
This project uses wandb for logging. To log in, run the following command and follow the instructions.
```bash
wandb login
```

## Features
* Automatic data preparation via huggingface datasets
* Stuctured codebase organized by PyTorch Lightning
* Pretrained model based on Prithvi-100M
* Training and evaluation visualization via wandb

## Training
* set the training data path in ```setting.json```
* Run train.py

## Inference
* Use inference.ipynb
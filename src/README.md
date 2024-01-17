# The Flintstones - Face Recognition

## Introduction
The entry point of the project is the `run.py` file. It is responsible for parsing the command line arguments and calling the appropriate functions for task 1 and task 2. 

## Description
The `run.py` contains a class called `FaceRecognizer` that does the following:
- Loads the dataset
- Loads the patch shapes
- Loads the model for localization (task 1)
- Loads the model for classification (task 2)
- (Optional) Saves the results in pickle files
- (Optional) Checks the results with the ground truth

Localization consists of two stages: 
- performing a `sliding window` over the image and classifying each patch as face or non-face
- performing `post-processing` on the results of the previous stage in order to obtain the `bounding boxes` of the faces and the corresponding confidence `scores`

Classification consists of one stage:
- classifying each face bounding box from the previous stage as one of the 5 classes
(`Barney`, `Betty`, `Fred`, `Wilma`, `Unknown`)


## Dependencies
To run the project you need to have installed the following packages:
- `numpy==1.24.3`
- `opencv-python==4.8.1.78`
- `torch==2.0.1`
- `torchvision==0.15.2`
- `pickle`
- `matplotlib==3.8.2`
- `tqdm==4.65.0`

You can install these dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run `run.py` you need to provide the following arguments:
- `-d` or `--dataset_path`(required): the path to the dataset
- `-p` or `--patch_shapes_path`(required): the path to the patch shapes
- `-ml` or `--model_localization_path`(required): the path to the model for localization
- `-mc` or `--model_classification_path`(required): the path to the model for classification
- `-c` or `--process_config_path`(required): the path to the configuration file for the post-processing
- `-s` or `--save_results`(optional): whether to save the results or not
- `-g` or `--ground_truth_path`(optional): the path to the ground truth file
- `-h` or `--help`: show the help message

Note that in the same directory as `run.py` there are the following files:
- `model_localization.pth`: the model for localization (task 1)
- `model_classification.pth`: the model for classification (task 2)
- `process_config.pkl`: the configuration file for the post-processing
- `patches.pkl`: the patch shapes

### Examples

This is the full command for running the project on task 1 and task 2:
```bash
python3 run.py -d ../validare/validare/ -p ./patches.pkl -ml ./model_localization.pth -mc ./model_classification.pth -c ./process_config.pkl -s ../evaluare/fisiere_solutie/ -g ../validare/
```

This is the full command for running the project on bonus:
```bash
python3 run.py -d ../validare/validare/ -m ./model_resnet.pth -s ../evaluare/fisiere_solutie/ -g ../validare/
```
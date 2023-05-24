# Deep Learning Project


## Dependencies

The list of dependencies can be found in the [`requirements.txt`](./requirements.txt) file. To install the dependencies, run the following command in your command line interface:

```
pip install -r requirements.txt
```

## Structure

```txt
Deep-Learning-Project
├── data
│   └── mock_test
├── src
│   ├── custom_datasets
│   │   └── cnn_custom_datasets.py
│   ├── inference
│   │   └── resnet_inference.py
│   ├── models
│   │   ├── alexnet_model.py
│   │   ├── base_model.py
│   │   ├── bow_model.py
│   │   ├── lstm_model.py
│   │   ├── resnet_model.py
│   │   └── transformer_model.py
│   ├── optimisers
│   │   ├── base_optimiser.py
│   │   ├── bow_classifier_optimiser.py
│   │   ├── lstm_classsifier_optimiser.py
│   │   └── pretrained_optimiser.py
│   ├── trainers
│   │   ├── base_trainer.py
│   │   ├── bow_classifier_trainer.py
│   │   ├── cnn_trainer.py
│   │   └── lstm_classifier_trainer.py
│   ├── utils
│   │   ├── cnn_utils.py
│   │   ├── dataset_loader.py
│   │   ├── definitions.py
│   │   ├── hyperparam_utils.py
│   │   ├── results_utils.py
│   │   └── text_preprocessor.py
│   └── main.py
├── tests
├── trained_models
│   └── resnet_pretrained.pth
├── MobileNet.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Code

The code can be run by executing the following command in your command line interface:

```
python ./src/main.py
```

## Contributions

- Thomas Bandy
    - ResNet
    - AlexNet
    - Inference script
- Alexander Budden
    - VGGNet
    - LeNet
- Panhanith Sokha
    - MobileNet    
- Sebastian Hadley
    - Experimentation with BOW model
    - Tested best performing LSTM model and best performing BOW model on test set.
- Jaydon Cameron
    - Experimentation with LSTM model
    - Fine-tuning pretrained models

A full list of commits to the repository can be found in [`git.log`](./git.log) or at
https://github.com/Jadocee/Deep-Learning-Project/commits/main.

## Intel Image Classification Training

This repository contains a set of Python classes to load, preprocess, and train a ResNet18 model on the Intel Image Classification dataset.

### Running the Inference Script

To run the inference script and generate predictions for individual images, follow the steps below:

1. Place all test images in the `test` folder located at `data/test`. Ensure that the images are in a format supported by the script (e.g., JPEG).
2. Download all the necessary dependencies by using the `requirements.txt` file. Install the dependencies by running the following command in your command line interface:

   ```
   pip install -r requirements.txt
   ```

   This will install all the required packages and libraries needed to run the script.

3. Once you have placed the test images and installed the dependencies, you can run the inference script. The script is located at `src/utils/inference_script.py`. Run the script by executing the following command:

   ```
   python ./src/inference/inference_script.py
   ```

   This command will execute the script and perform inference on the test images using a pretrained model.

4. After running the script, the results for each individual image will be recorded in the `preds.csv` file, which can be found in the `test` folder. The `preds.csv` file will contain the predicted labels for the test images.

By following these steps, you will be able to run the inference script and generate predictions for the test images. Make sure to verify that the test images are placed correctly and that the script has the necessary permissions to access the files and directories it requires.

### Table of Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [References](#references)

### Files

1. `dataset.py`: Contains the `Dataset` class, which loads images and labels from a directory.
2. `resnet.py`: Contains the implementation of the `ResNet18` model and its components.
3. `train.py`: Contains the `Train` class, which handles the preparation of the data and training the model.
4. `main.py`: The main script that uses the above classes to train the model.

### Dependencies

This project uses the following libraries:

- PyTorch
- torchvision
- pandas
- PIL
- matplotlib

Please ensure these are installed before running the scripts.

### Dataset

The dataset used is the Intel Image Classification dataset, which contains 6 categories: 'buildings', 'forest', 'glacier', 'mountain', 'sea', and 'street'. The dataset is expected to be in a specific directory structure as specified in the `Dataset` class in `dataset.py`.

### Model

The model used is a ResNet18 model implemented in the `ResNet18` class in `resnet.py`.

### Training

The model is trained using the Adam optimizer with a learning rate of 0.001 and the Cross Entropy Loss function. Training is performed for a specified number of epochs (10 by default).

### References

- Kaiming He, et al. "Deep Residual Learning for Image Recognition." arXiv:1512.03385
- https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
- https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
- All Docstrings were generated by ChatGPT.




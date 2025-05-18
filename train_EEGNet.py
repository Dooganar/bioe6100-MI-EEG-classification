import mne

import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split

import json

import EEGNet

DATA_DIR = "/home/reuben/Documents/eeg-data/"
MODELS_DIR = "./models/"

def load(load_path):
    epochs = mne.read_epochs(load_path)
    data = epochs.get_data(copy=True)
    labels = epochs.events[:,-1]
    return data, labels

def train(name, load_path, epochs):
    # Choosing Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    data, labels = load(load_path)

    # Normalizing Labels to [0, 1, 2]
    y = labels - np.min(labels)

    # Normalizing Input features: z-score(mean=0, std=1)
    X = (data - np.mean(data)) / np.std(data)

    # Spliting  Data: 90% for Train and 10% for Test
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Converting to Tensor
    X_train = torch.Tensor(X_train).unsqueeze(1).to(device)
    X_test = torch.Tensor(X_test_).unsqueeze(1).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test_).to(device)

    # Creating Tensor Dataset
    train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)

    # Printing the sizes
    print("Size of X_train:", X_train.size())
    print("Size of X_test:", X_test.size())
    print("Size of y_train:", y_train.size())
    print("Size of y_test:", y_test.size())

    print(y_train)
    print(y_test)

    ## MODEL SUMMARY

    # input_size = (1, 64, 961)
    # eegnet_model = EEGNet.EEGNetModel().to(device)
    # summary(eegnet_model, input_size)

    chans = X_train.size()[2]
    time_points = X_train.size()[3]

    eegnet_model = EEGNet.EEGNetModel(chans=chans, time_points=time_points).to(device)

    # Training Hyperparameters
    EPOCHS = epochs
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    trainer = EEGNet.TrainModel()
    trained_eegnet_model, train_info = trainer.train_model(eegnet_model, train_dataset, learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE, epochs=EPOCHS)
    torch.save(trained_eegnet_model.state_dict(), MODELS_DIR + name + '.pth')
    
    train_metas = [train_info, X_test_.tolist(), y_test_.tolist(), chans, time_points]

    with open(MODELS_DIR + name + ".json", 'w') as json_file1:
        json.dump(train_metas, json_file1)

def evaluate(name):
    with open(MODELS_DIR + name + ".json", 'r') as json_file1:
        train_metas_loaded = json.load(json_file1)

    train_info, X_test_, y_test_, chans, time_points = train_metas_loaded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = torch.Tensor(X_test_).unsqueeze(1).to(device)
    y_test = torch.LongTensor(y_test_).to(device)

    test_dataset = TensorDataset(X_test, y_test)

    trained_eegnet_model = EEGNet.EEGNetModel(chans=chans, time_points=time_points).to(device)
    trained_eegnet_model.load_state_dict(torch.load(MODELS_DIR + name + '.pth', map_location=torch.device('cpu')))
    trained_eegnet_model.eval()
    classes_list = ['rest', 'hands', 'feet']
    eval_model = EEGNet.EvalModel(trained_eegnet_model, MODELS_DIR + name)
    test_accuracy = eval_model.test_model(test_dataset)
    eval_model.plot_confusion_matrix(test_dataset, classes_list)

    plt.plot(train_info[0], train_info[1], label="Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(MODELS_DIR + name + '_loss_curve.png')
    plt.show()

    plt.plot(train_info[0], train_info[2], label="Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(MODELS_DIR + name + '_accuracy_curve.png')
    plt.show()

if __name__ == "__main__":
    print("Running 'train_EEGNet.py' directly")
    
    for i in range(1, 25):
        name = "task1_s" + str(i)
        load_path = DATA_DIR + "/physionet-fifs/task1/s" + str(i) + "-epo.fif"
        train(name, load_path, epochs=200)
        # evaluate(name)
import mne

import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd

import json
import csv

import EEGNet

DATA_DIR = "/home/reuben/Documents/eeg-data/"
MODELS_DIR = "./models/"

def load(load_path):
    epochs = mne.read_epochs(load_path)
    data = epochs.get_data(copy=True)
    labels = epochs.events[:,-1]
    return data, labels

def train(name, load_path, save_path_folder, hypers):
    epochs = hypers["epochs"]
    test_ratio = hypers["test-ratio"]

    # Choosing Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, labels = load(load_path)

    # Normalizing Labels to [0, 1, 2]
    y = labels - np.min(labels)

    # Normalizing Input features: z-score(mean=0, std=1)
    X = (data - np.mean(data)) / np.std(data)

    # ------------------Consider Class Imbalances-------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    
    number_of_classes = len(np.unique(y))
    print("number_of_classes:", number_of_classes)
    class_counts = []
    for i in range(number_of_classes):
        class_counts.append(y.tolist().count(i))
        print(f"class_counts[{i}] =", class_counts[i], "| Weight =", class_weights[i])

    # Loss Function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    # ---------------------------------------------------------------

    # Spliting  Data: 90% for Train and 10% for Test
    X_train, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)

    # Converting to Tensor
    X_train = torch.Tensor(X_train).unsqueeze(1).to(device)
    X_test = torch.Tensor(X_test_).unsqueeze(1).to(device)
    y_train = torch.LongTensor(y_train_).to(device)
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
    trained_eegnet_model, train_info = trainer.train_model(eegnet_model, train_dataset, criterion, learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE, epochs=EPOCHS)
    torch.save(trained_eegnet_model.state_dict(), save_path_folder + name + '.pth')
    
    train_metas = [train_info, X_test_.tolist(), y_test_.tolist(), y_train_.tolist(), chans, time_points, class_counts]

    with open(save_path_folder + name + ".json", 'w') as json_file1:
        json.dump(train_metas, json_file1)

def evaluate(name, saved_path_folder, pltshow=False, save=True, verbose=False):
    saved_path = saved_path_folder + name + ".pth"
    with open(saved_path_folder + name + ".json", 'r') as json_file1:
        train_metas_loaded = json.load(json_file1)

    train_info, X_test_, y_test_, y_train_, chans, time_points, class_counts = train_metas_loaded

    if verbose: 
        print("-------- Evaluating", name, "-----------")
        print("y_train:", y_train_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = torch.Tensor(X_test_).unsqueeze(1).to(device)
    y_test = torch.LongTensor(y_test_).to(device)

    test_dataset = TensorDataset(X_test, y_test)

    trained_eegnet_model = EEGNet.EEGNetModel(chans=chans, time_points=time_points).to(device)
    trained_eegnet_model.load_state_dict(torch.load(saved_path, map_location=torch.device('cpu')))
    trained_eegnet_model.eval()
    classes_list = ['rest', 'hands', 'feet']
    eval_model = EEGNet.EvalModel(trained_eegnet_model, saved_path_folder + name)
    test_accuracy = eval_model.test_model(test_dataset)
    eval_model.plot_confusion_matrix(test_dataset, classes_list, pltshow=pltshow, save=save)

    fig = plt.figure()
    plt.plot(train_info[0], train_info[1], label="Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save: plt.savefig(save_path_folder + name + '_loss_curve.png')
    if pltshow: plt.show() 
    else: plt.close(fig)

    fig = plt.figure()
    plt.plot(train_info[0], train_info[2], label="Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if save: plt.savefig(save_path_folder + name + '_accuracy_curve.png')
    if pltshow: plt.show()
    else: plt.close(fig)

    return test_accuracy

def batch_train(task, subject_range, load_path_folder, save_path_folder, hypers):
    for i in range(subject_range[0], subject_range[1]):
        name = "task"+ str(task) + "_s" + str(i)
        load_path = load_path_folder + "s" + str(i) + "-epo.fif"
        train(name, load_path, save_path_folder, hypers)

def batch_evaluate(name, subject_range, saved_path_folder):
    accuracy_datas = [
        ["Subject", "Task 1", "Task 2"]
    ]

    for i in range(subject_range[0], subject_range[1]):
        print("Evaluating Model Performance On Subject", i)
        name_task1 = "task1_s" + str(i)
        name_task2 = "task2_s" + str(i)

        test_accuracy_task_1 = evaluate(name_task1, saved_path_folder)
        test_accuracy_task_2 = evaluate(name_task2, saved_path_folder)
        accuracy_data = [i, test_accuracy_task_1, test_accuracy_task_2]
        accuracy_datas.append(accuracy_data)

    # File path for the CSV file
    csv_file_path = saved_path_folder + "/" + name + '-test-accuracys.csv'

    # Open the file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a csv.writer object
        writer = csv.writer(file)
        # Write data to the CSV file
        writer.writerows(accuracy_datas)

    # Print a confirmation message
    print(f"CSV file '{csv_file_path}' created successfully.")

if __name__ == "__main__":
    print("Running 'train_EEGNet.py' directly")

    hyperparameters = {
        "epochs": 200,
        "test-ratio": 0.3
    }

    # name = "task1_s1"
    # saved_path_folder = MODELS_DIR + "physionet-8-channel/"

    # train(name, load_path, hyperparameters)
    # evaluate(name, saved_path_folder, pltshow=True, save=False, verbose=True)
    

    task = 1
    load_path_folder = DATA_DIR + "/physionet-fifs-8-channel/task"+ str(task) +"/"
    save_path_folder = MODELS_DIR + "/physionet-8-channel/"
    batch_train(task, [1, 110], load_path_folder, save_path_folder, hyperparameters)

    # batch_evaluate("models-8ch-tasks12-200epoch", [1, 25], save_path_folder)
    
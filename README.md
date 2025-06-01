# bioe6100-MI-EEG-classification


# EEGNet Classification of Motor Movement vs. Motor Imagery

This repository contains code, data processing scripts, and analysis tools used for evaluating EEGNet classification performance on EEG datasets recorded during motor movement (MM) and motor imagery (MI) tasks. This work was conducted as part of the final project for BIOE6100 - Fundamentals of Neuroengineering at The University of Queensland.

## Project Overview

Motor imagery (MI) is a promising paradigm for brain–computer interfaces (BCIs), but often suffers from weaker and less consistent EEG signals than actual motor movement (MM). This project investigates how effectively EEGNet—a compact convolutional neural network (CNN)—can classify MM and MI EEG data, and how classification performance varies depending on electrode density and signal quality.

Two datasets are used:
- **Dataset A**: 64-channel wet electrode EEG data from 109 subjects (PhysioNet).
- **Dataset B**: 8-channel dry electrode EEG data collected from a single subject using OpenBCI hardware.

## Key Findings

- EEGNet performs significantly better on MM than MI data for 64-channel recordings (mean +3.74% accuracy, *p = 0.0071*).
- This performance gap disappears for 8-channel configurations on the same dataset.
- For the OpenBCI dataset (Dataset B), MM classification achieved 87.5% accuracy vs. 56.25% for MI—a substantial gap likely due to dry electrode limitations and subject variability (there was only one subject for Dataset B - me).

Full results and discussion can be found in the [final report PDF](./EEGNet-MM-vs-MI-report-v1-1.pdf).

## Repository Structure

```
bioe6100-MI-EEG-classification/
├── results/
│   ├── models-64ch-tasks12-200epoch-test-accuracys.csv
│   ├── models-8ch-tasks12-200epoch-test-accuracys.csv
│   └── results_analysis.py
├── src/
│   ├── eegnet.py                   # EEGNet architecture and training class (PyTorch)
│   ├── train_eegnet.py             # Script for model training and evaluation
│   ├── proces_physionet_data.py    # Preprocessing pipeline for Dataset A
│   ├── collect_openbci_data.py     # EEG acquisition script using BrainFlow (Dataset B)
│   └── process_openbci_data.py     # Preprocessing pipeline for Dataset B
└── README.md
```


## Acknowledgements

* PhysioNet: EEG Motor Movement/Imagery Dataset
* OpenBCI and BrainFlow
* Lawhern et al. (2018): EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces


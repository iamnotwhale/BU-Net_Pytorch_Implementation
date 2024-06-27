import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
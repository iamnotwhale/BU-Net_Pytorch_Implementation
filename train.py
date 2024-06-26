from data_loader import Custom2DBraTSDataset
from model import UNet, BU_net, Unet_WC, BU_Net_Loss, dice_score, accuracy_check, accuracy_check_for_batch
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


def train_model(trainloader, model, optimizer, device):
    model.train()
    loss_criterion = BU_Net_Loss()  # Instantiate the loss object here
    loss_criterion.to(device)       # Ensure the loss model is on the correct device
    for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = inputs.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        inputs = inputs.float()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)  # Correct usage of the loss function
        loss.backward()
        optimizer.step()

def val_model(model, valloader, loss, device):

    total_val_loss = 0
    total_val_acc = 0
    n=0

    for batch, (inputs, labels) in tqdm(enumerate(valloader), total = len(valloader)):
        with torch.no_grad():

            inputs = inputs.to(device)
            labels = labels.to(device=device, dtype=torch.int64)

            outputs = model(inputs)
            loss = BU_Net_Loss()
            loss_value = loss(outputs, labels)

            outputs = np.transpose(outputs.cpu(), (0, 2, 3, 1))
            preds = torch.argmax(outputs, dim=3).float()

            acc = accuracy_check_for_batch(labels.cpu(), preds.cpu(), inputs.size()[0])
            total_val_acc += acc
            total_val_loss += loss_value.cpu().item()

    return total_val_acc/(batch+1), total_val_loss/(batch+1)

def main():
    pass
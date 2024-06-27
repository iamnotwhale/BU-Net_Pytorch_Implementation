import argparse
import os
from data_loader import Custom2DBraTSDataset
from model import UNet, BU_net, Unet_WC, BU_Net_Loss, dice_score, accuracy_check, accuracy_check_for_batch, get_loss_train
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

def get_loss_train(model, trainloader, loss, device):
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (inputs, labels) in tqdm(enumerate(trainloader), total = len(trainloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device = device, dtype = torch.long)
            inputs = inputs.float()

            outputs = model(inputs)
            loss = BU_Net_Loss()
            loss_val = loss(outputs, labels)
            outputs = np.transpose(outputs.cpu(), (0,2,3,1))
            preds = torch.argmax(outputs, dim=3).float()
            acc = accuracy_check_for_batch(labels.cpu(), preds.cpu(), inputs.size()[0])
            total_acc += acc
            total_loss += loss_val.cpu().item()

    return total_acc/(batch+1), total_loss/(batch+1)

def main(epochs, model, device, trainloader, valloader, optimizer):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_model(trainloader, model, optimizer, device)
        train_acc, train_loss = get_loss_train(model, trainloader, BU_Net_Loss, device)
        print("epoch", epoch + 1, "train loss : ", train_loss, "train acc : ", train_acc)

        val_acc, val_loss = val_model(model, valloader, BU_Net_Loss, device)
        print("epoch", epoch + 1, "val loss : ", val_loss, "val acc : ", val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'./{str(epoch)}.pth')

    return history

if __name__ == '__main__':

    torch.manual_seed(123)

    args = argparse.ArgumentParser()

    args.add_argument('--dataset_folder', type=str, required=True, help='Path to the training data folder')
    args.add_argument('--modality', type=str, default='t1', help='Modality of the data to train on, You can choose either t1 or t2 or t1ce or flair')
    args.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    args.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    args.add_argument('--model', type=str, default='Unet', help='You can choose either Unet or Unet_WC')
    args.add_argument('--device', type=str, default='cpu', help='Device to train the model on')

    args = args.parse_args()

    device = torch.device(args.device)
    
    if args.model == 'Unet':
        model = UNet()
    elif args.model == 'Unet_WC':
        model = Unet_WC()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t1_dataset = Custom2DBraTSDataset(args.dataset_folder, modality='t1', num_slices=5)
    t2_dataset = Custom2DBraTSDataset(args.dataset_folder, modality='t2', num_slices=5)
    t1ce_dataset = Custom2DBraTSDataset(args.dataset_folder, modality='t1ce', num_slices=5)
    flair_dataset = Custom2DBraTSDataset(args.dataset_folder, modality='flair', num_slices=5)

    dataset = torch.utils.data.ConcatDataset([t1_dataset, t2_dataset, t1ce_dataset, flair_dataset])

    trainset, valset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    history = main(args.epochs, model, device, trainloader, valloader, optimizer)
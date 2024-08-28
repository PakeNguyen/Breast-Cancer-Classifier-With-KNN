import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
import argparse
import shutil
import matplotlib.pyplot as plt
from CNN_Model import SimpleCNN
from torchvision.models import resnet34 , ResNet34_Weights
from data_loader_ANIMAL import AnimalDataSet
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def get_args():
        parse = argparse.ArgumentParser()
        parse.add_argument("--data_path", "-d", type=str, default="Khoa3_DeepLN_CV_coban/data_xaydungclass/Animal_V2/animals")
        parse.add_argument("--epochs", "-e", type=int, default=100)
        parse.add_argument("--batch_size", "-b", type=int, default=16)
        parse.add_argument("--image_size", "-i", type=int, default=224)
        parse.add_argument("--lr", "-l", type=float, default=1e-2)    
        parse.add_argument("--log_path", "-p", type=str, default="Khoa3_DeepLN_CV_coban/tensorboard/animals")
        parse.add_argument("--checkpoint_path", "-c", type=str, default="Khoa3_DeepLN_CV_coban/checkpoint/animals")
        args = parse.parse_args()

        return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10)
    model.to(device)

    transforms = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        Resize((args.image_size, args.image_size))
    ]) 
    train_dataset = AnimalDataSet(root=args.data_path, is_train=True, transforms = transforms)
    Train_DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False
    )

    test_dataset = AnimalDataSet(root=args.data_path, is_train=False, transforms=transforms)
    Test_DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False
    )

    criterion = nn.CrossEntropyLoss()
    otimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if os.path.isdir(args.log_path):
         shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
         os.makedirs(args.checkpoint_path)
    
    write = SummaryWriter(args.log_path)
    best_accuracy = 0

    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        progress_Bar = tqdm(Train_DataLoader, colour="cyan")
        for i, (images, labels) in enumerate(progress_Bar):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            progress_Bar.set_description(f"Epochs {epoch}/{args.epochs}. Loss {loss:0.4f}")
            write.add_scalar("Train/Loss", loss, epoch*(len(Train_DataLoader)) + i)
            otimizer.zero_grad()
            loss.backward()
            otimizer.step()

        # VALIDATION
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():  
            for images, labels in Test_DataLoader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                predictions = torch.argmax(output, dim=1)
                
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
            Accuracy = accuracy_score(all_labels, all_predictions)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            loss_mean = np.mean(all_losses)
            
            print(f"Epochs {epoch}/{args.epochs}. Loss {loss_mean}. Accuracy {Accuracy}")
            write.add_scalar("Test/Loss", loss_mean, epoch)
            write.add_scalar("Test/Accuracy", Accuracy, epoch)

            checkpoint = {
                 "model_state_dict" : model.state_dict(),
                 "epoch" : epoch,
                 "optimizer_state_dict" : otimizer.state_dict()
            }

            torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
            if Accuracy > best_accuracy:
                 torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
                 best_accuracy = Accuracy

if __name__ == "__main__":
    args = get_args()
    train(args)
    
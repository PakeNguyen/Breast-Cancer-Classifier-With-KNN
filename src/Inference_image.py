import os

import numpy as np
from CNN_model_B9 import SimpleCNN
import torch
import torch.nn as nn
from LuyenTapDataSet_Animal.luyentap_B5_Animal_1 import AnimalDataSet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose,ToTensor,Resize, Normalize
from torchvision.models import resnet34 , ResNet34_Weights
import argparse
from tqdm.autonotebook import tqdm # Thêm autonotebook thì chạy trên colab hoặc jupyter notebook sẽ ít bị lỗi
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import cv2 
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Train NN model")
    parser.add_argument("--image_path", "-p", type=str, default='image.jpg')
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/animals")
    args = parser.parse_args()

    return args

    

def train(args):
    model = resnet34()
    model.fc = nn.Linear(in_features=512, out_features=11)
    classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(agrs.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # bước này chuyển sang RGB là loại ảnh PIL lý do: ToTensor chỉ làm việc với PIL
    image = cv2.resize(image, (args.image_size,args.image_size))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # equivalent to ToTensor() from Pytorch 
    image = image / 255.
    
    # equivalent to normalize
    image = (image - mean)/std
    image = np.transpose(image, (2,0,1))[None, :, :, :] # 2 bước này tương đương với hàm ToTensor
    image = torch.from_numpy(image).float().to(device) # Bước này chuyển từ array sang tensor 
    
    checkpoint = torch.load(os.path.join(args.checkpoint_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()# gọi hàm để cho output không có số âm
    with torch.no_grad():
        output = model(image)
        prob = softmax(output) #dòng này để cho output không có số âm nằm trong 0-1 để xuống dưới tính mới được 
        
        # Dòng này lấy ra vị trí max trong tensor
        predicted_prob, predicted_class = torch.max(prob,dim=1) # [0] nghĩa là chỉ lấy số bỏ cái ngoặc list []
        score = predicted_prob[0] * 100
        cv2.imshow("Day la con {} - Do chinh xac : {:0.2f}%".format(classes[predicted_class[0]],score), cv2.imread(agrs.image_path))
        cv2.waitKey(0)

    # transforms = Compose([
    #     ToTensor(),
    #     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     Resize((args.image_size, args.image_size))
    # ])
    



if __name__ == '__main__':
    agrs = get_args()
    train(agrs)



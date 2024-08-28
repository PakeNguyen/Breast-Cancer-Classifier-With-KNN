from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import Resize, Compose,ToTensor
from PIL import Image


class AnimalDataSet(Dataset):
    def __init__(self,root,is_train,transforms=None):
        if is_train:
            data_path = os.path.join(root,"train")
        else:
            data_path = os.path.join(root,"test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", 
                           "horse", "sheep", "spider", "squirrel"]
        
        self.image_paths = []
        self.labels = []

        for index,category in enumerate(self.categories):
            subdir_path = os.path.join(data_path, category)
            for file_name in os.listdir(subdir_path):
                data = os.path.join(subdir_path,file_name) 
                # dòng dưới sẽ lưu cả chục , trăm nghìn tấm ảnh rất tốn memory và có khi ko chứa nổi
                # image = cv2.imread(data)
                # self.images.append(image)
                self.image_paths.append(data)
                # vì vậy ta chỉ lưu đường dẫn string của ảnh sẽ tiết kiệm rất nhiều memory cả triệu đường dẫn cũng đc
                self.labels.append(index) 
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_paths[index])
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label

if __name__ == "__main__":
    transforms = Compose([
        ToTensor(),
        Resize((240,240))
    ])
    Train_Data = AnimalDataSet(root="Khoa3_DeepLN_CV_coban/data_xaydungclass/Animal_V2/animals",is_train=True,transforms=transforms)
    # image,label= Train_Data[2000]
    
    train_Dataloader = DataLoader(
        dataset = Train_Data,
        batch_size=8,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    for images, labels in train_Dataloader:
        print(images.shape , labels.shape)
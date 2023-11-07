import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None,dataL=-1):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)
        self.ext1=os.path.splitext(self.images[0])[1];
        if dataL>0:
            self.images=self.images[:dataL]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.image_dir, self.images[index])
        mask_path=os.path.join(self.mask_dir, self.images[index].replace(self.ext1, ".png"))

        image=np.array(Image.open(img_path).convert("RGB"),dtype=np.float32)
        mask=np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask=(mask-np.min(mask))/(np.max(mask)-np.min(mask)+10**-5)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        mask[mask<0.5]=0.; mask[mask>=0.5]=1.0
        
        a1,a2,a3=image.shape
        if a1>a2:
            image=image.transpose((1,0,2))
            mask=mask.transpose((1,0))
        if self.transform is not None:
            augmentations= self.transform(image=image,mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
            
        return image,mask
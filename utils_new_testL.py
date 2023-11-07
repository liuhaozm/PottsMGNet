import torch
import torchvision
from dataset_new import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint,model):
    print("=> Loadinig checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    trainL=-1,
    testL=-1,
    num_workers=4,
    pin_memory=False,
    ):
    train_ds=ImageDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        dataL=trainL,
    )
    
    train_loader=DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_ds=ImageDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        dataL=testL,
    )
    
    val_loader=DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return train_loader, val_loader


def check_accuracy(loader,model, device="cuda"):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=model(x)

            preds=(preds>0.5).float()
            num_correct+=(preds==y).sum()
            num_pixels += torch.numel(preds)
            dice_score+=(2*(preds*y).sum())/((preds+y).sum()+1e-8)
    
    acc=num_correct/num_pixels*100
    dice=dice_score/len(loader)

    
    model.train()
    return acc,dice
    
def save_predictions_as_imgs(
    loader, model,folder="saved_images/", device="cuda"):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.to(device=device)
        with torch.no_grad():

            preds=model(x)
            preds=(preds>0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png")
        
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        
    model.train()
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    
    
def save_compare_as_imgs(
    loader, model,folder="saved_images/", batch_size=8, device="cuda"):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.to(device=device)
        y=y.to(device=device)
        with torch.no_grad():

            preds=model(x)
            preds=(preds>0.5).float()
            imgrid0=make_grid(x,batch_size)
            imgrid1=make_grid(y.unsqueeze(1),batch_size)#.unsqueeze(0)
            imgrid2=make_grid(preds,batch_size)
            imgrid3=make_grid(x*preds,batch_size)
            imgrid=make_grid(torch.stack((imgrid0,imgrid1,imgrid2,imgrid3),dim=0),1)

        
            
        torchvision.utils.save_image(
            imgrid, f"{folder}/batch_{idx}.png")
        

        
    model.train()
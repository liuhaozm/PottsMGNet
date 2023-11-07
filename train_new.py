

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim


from PottsMGNet_model import POTTSNET


from utils_new_testL import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_compare_as_imgs,
)

fname='taiv4varyingNoise'
taskNo='2654'






learning_rate=1e-4
device="cuda" if torch.cuda.is_available() else "cpu"
# hyper parameters
batch_size=8
num_epochs=500
num_workers=0
image_height=128 
image_width=192 
trainL=800 # training data size, use -1 for all data
testL=200 # training data size, use -1 for all data
pin_memory=False
load_model=False
train_dir="../../data/train" # dictionary of training image
train_maskdir="../../data/train_mask" # dictionary of training mask
val_dir="../../data/test" # dictionary of testing image
val_maskdir="../../data/test_mask" # dictionary of testing mask
#


class pp:
    in_channels=4
    mid_channels=[32, 32, 64, 128, 256]

    times_list=[3, 3, 3, 5, 5]
    tau=0.5 # times step size
    cnsts=[1., 40.] # [epsilon* tau, lambda/epsilon]. The result can be sensitive to epsilon* tau. The larger it is, the smoother the result is. Set it between 0.7 and 1 gives good results. 
    num_blocks=4 # number of blocks
    
    
    epsilon=cnsts[0]/tau
    lambdaa=epsilon*cnsts[1] # weight of the length penalty term

    sigma=0.5 # sd of the Gaussian kernel
    iter_num=1 # number of fixed iteration
    kernel_size_bound=5 # largest kernel size allowed
    connect=True # True if use skip-connections between encoder and decoder
    BNLearn=True # True if learn paramters in batch normalization
    device="cuda" if torch.cuda.is_available() else "cpu"
    alpha=1 # relaxation ration in fixed point iteration. Changing this parameter may lead unstability of the method.
    
    lambdaLearn=False #True if we allow lambda to be learnable
    timevarying=True # True if we allow wegiths are different among blocks
    
    tau_explicit=True # True if we explicitly use time evolution. False if absorb linear terms into convolution
    
args=pp





def train_fn(loader,model,optimizer,loss_fn):
#     loop=tqdm(loader)
    start=time.time()
    losses=[]
    nums=[]
    for batch_idx, (data,targets) in enumerate(loader):
        data=data.to(device=device)
        targets=targets.float().unsqueeze(1).to(device=device)
        

        predictions=model(data)
        loss=loss_fn(predictions,targets)
        losses.append(loss.item())
        nums.append(len(data))
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
       
    end=time.time()
    tt=end-start
    total=np.sum(nums)
    avg_loss=np.sum(np.multiply(losses,nums))/total
    return tt, avg_loss
        
def main():
    print('filename= ', fname)

    print('mid_channels= ',args.mid_channels)
    print('times_list= ',args.times_list)
    print('cnsts=', args.cnsts)
    print('lambdaa= ',args.cnsts[0]*args.cnsts[1]/args.tau)
    print('tau= ',args.tau)
    print('epsilon= ',args.cnsts[0]/args.tau)
    print('sigma= ',args.sigma)
    print('num_blocks= ',args.num_blocks)
    print('M= ',args.M)
    print('iter_num= ',args.iter_num)
    
    print('skip_connect= ',args.connect)
    print('BNLearn= ',args.BNLearn)
    print('lambdaLearn= ',args.lambdaLearn)
    print('timevarying= ',args.timevarying)
    print('tau*eps= ', args.cnsts[0])
    print('lambda/eps= ', args.cnsts[1])
    print('tau_explicit= ', args.tau_explicit)
    
    lossAll=[]
    accAll=[]
    diceAll=[]
    traintime=[]

    model=POTTSNET(args).to(device)
    
    loss_fn=nn.BCELoss()

    optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    

    
    if load_model:
        checkpoint_old=torch.load(loadname)
        model.load_state_dict(checkpoint_old["state_dict"])
        optimizer.load_state_dict(checkpoint_old['optimizer'])
        accAll=checkpoint_old['accAll']
        lossAll=checkpoint_old['lossAll']
        diceAll=checkpoint_old['diceAll']
        traintime=checkpoint_old['traintime']
        acc_old=checkpoint_old['acc_old']
        dice_old=checkpoint_old['dice_old']
    else:
        acc_old=0
        dice_old=0

    print('acc_old= ',acc_old)
    print('dice_old= ',dice_old)

    loss=10. 
    epoch=0
    
    # progressive training
    for noise_sd in (0.,0.3,0.5,0.8,1.):

        savename=fname+'_sd0'+str(int(noise_sd*10))+".pth.tar"
        train_transform =A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            A.GaussNoise(var_limit=(0,(noise_sd)**2),always_apply=True,p=1),
            ToTensorV2(),
        ])

        val_transform=A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ])

        train_loader, val_loader=get_loaders(
            train_dir,
            train_maskdir,
            val_dir,
            val_maskdir,
            batch_size,
            train_transform,
            val_transform,
            trainL,
            testL,
            num_workers,
            pin_memory,
        )        
        for epoch in range(num_epochs):
            tt,loss=train_fn(train_loader, model, optimizer, loss_fn)

            lossAll.append(loss)

            # check accuracy
            acc,dice=check_accuracy(val_loader, model,device=device)


            accAll.append(acc)
            diceAll.append(dice)
            traintime.append(tt)

            flag=(acc>acc_old)

            if flag:

                acc_old=acc
                dice_old=dice

            print(
                   f"noise: {noise_sd}, Epoch  {epoch}/{num_epochs}, loss: {loss:.4f}, dice: {dice:.4f}, acc: {acc:.2f}, **acc: {acc_old:.2f}, time used: {tt:.2f}s")



        # save model
        checkpoint={
            "parameters":args,
            "times_list":args.times_list,
            "mid_channels":args.mid_channels,
            "num_blocks": args.num_blocks,
            "lambdaa": args.lambdaa,
            "tau": args.tau,
            "epsilon": args.epsilon,
#                 "M":args.M,
            "iter_num":args.iter_num,
#                 "iter_num_len":args.iter_num_len,
            "sigma":args.sigma,
            "kernel_size_bound":args.kernel_size_bound,
            "connect":args.connect,
            "BNLearn":args.BNLearn,
            "alpha":args.alpha,

            "lambdaLearn":args.lambdaLearn,
            "timevarying":args.timevarying,

            "tau_explicit":args.tau_explicit,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lossAll": lossAll,
            "accAll": accAll,
            "diceAll": diceAll,
            "acc_old": acc_old,
            "dice_old": dice_old,
            "traintime": traintime,
        }
        torch.save(checkpoint,savename)
   
      ###########print some examples
#             if epoch%200==0:
#                 save_compare_as_imgs(
#                     loader=val_loader,model=model,batch_size=batch_size,folder="saved_images/",device=device,
#                 )
    
if __name__ == "__main__":
    main()
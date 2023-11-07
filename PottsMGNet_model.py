



import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



def getb(s):
    b = 1./(1. + torch.exp(-10*s)) 
    return b

def GaussKernel(sigma=None, requires_grad=False):
# function create a Gaussian kernel.
# M: kernel size
# sigma: standard deviation

    M=max(4*int(2*sigma)+1,5)
    xx = torch.arange(0, M) - (M - 1.0) / 2.0
    grid_x =xx.reshape(M,1)*torch.ones(1,M)
    grid_y =torch.ones(M,1)*xx.reshape(1,M)
    d2=grid_x**2+grid_y**2
    w=torch.exp(-d2/(2*sigma**2))
    w=w.reshape((1,1,M,M))/w.sum()
    w.requires_grad=requires_grad
    return w

class sigAct1(nn.Module):
# activation function without length penalty
    def __init__(self,args=None):
        super().__init__()
        self.tau=args.tau # time step size
        self.epsilon=args.cnsts[0]/args.tau 
        self.iter_num=args.iter_num # iteration number of fixed point iteration
        self.sig=nn.Sigmoid()
        self.alpha=args.alpha # relaxiation ration in the fixed point iteration

    def forward(self,x):

        u=0.5
        for idx in range(abs(self.iter_num)):
            uu=(u-x)/self.tau/self.epsilon
            uu=self.sig(-uu)
            u=(1-self.alpha)*u+self.alpha*uu
        return u

class POTTS(nn.Module):
    def __init__(self,args):
        
        super(POTTS,self).__init__()
        self.sig=nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)

        self.tau=args.tau
        self.epsilon=args.cnsts[0]/args.tau
        self.iter_num=max(abs(args.iter_num),1)
        self.sigma=args.sigma
        self.device=args.device

        self.alpha=args.alpha
        self.lambdaa=args.cnsts[0]*args.cnsts[1]/args.tau

        self.mid_channels=args.mid_channels
        self.level_max=len(args.mid_channels)
        self.kernel_size_max=3**self.level_max
        self.skip_connection=args.connect
        self.times_list=args.times_list
        self.num_blocks=args.num_blocks
        self.kernel_size_bound=args.kernel_size_bound
        self.BNLearn=args.BNLearn
        
        self.tau_explicit=args.tau_explicit
        self.lambdaLearn=args.lambdaLearn
        # self.bLearn=args.bLearn

        if args.iter_num<0: 
            self.sigAct1 = self.sig
        elif args.iter_num==0: 
            self.sigAct1 = self.relu
        else:
            self.sigAct1=sigAct1(args=args)

class sigAct(POTTS):
    # activation function with length penalty
    # lambdaa: weight of length penalty
    # epsilon: parameter in sigmoid function
    # iter_num: number of fixed point iterations
    # alpha: relaxation step
    def __init__(self, args=None, in_channels=1):
        super().__init__(args=args)
        self.M=max(4*int(2*args.sigma)+1,5)
        self.in_channels=in_channels
        self.Gauss=GaussKernel(sigma=args.sigma).to(args.device)
        self.weight=self.Gauss.repeat(self.in_channels,1,1,1)
        self.convG=nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.M,
                             groups=self.in_channels, padding=self.M//2, bias=False)
        self.convG.weight.data=self.weight
        self.convG.weight.requires_grad=False
        
    def forward(self,x,c):
        u=self.lambdaa*getb(c)*self.convG(1.-2.*x)
        u=self.sig(-u/self.epsilon)
        
        for idx in range(max(abs(self.iter_num),1)):
            uu=(u-x)/self.tau+self.lambdaa*getb(c)*self.convG(1.-2.*u)
            uu=self.sig(-uu/self.epsilon)

            u=(1-self.alpha)*u+self.alpha*uu

        return u

class MGPCConv_first(POTTS):
# finest grid level on the left branch of the V-cycle
    def __init__(self, args=None, times=2, in_channels=1,out_channels=1,kernel_size=None):
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.times=times
        self.kernel_size=np.minimum(kernel_size,self.kernel_size_bound)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=in_channels+3,out_channels=out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
        self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
        self.model.append(self.sigAct1)
        for i in np.arange(self.times-1):
            self.model.append(nn.Conv2d(in_channels=out_channels,out_channels=out_channels, 
                      kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
            self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
            self.model.append(self.sigAct1)

    def forward(self,x,flist):
        out1=torch.cat((x,flist[0]),dim=1)
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](out1)+torch.sum(x,dim=1,keepdim=True)/self.in_channels
        else:
            out=self.model[0](out1)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1):           
            if self.tau_explicit:
                out=self.out_channels*self.tau*self.model[i*3+3](out)+torch.sum(out,dim=1,keepdim=True)/self.out_channels
            else:
                out=self.model[i*3+3](out)
            out=self.model[i*3+4](out)
            out=self.model[i*3+5](out)

        return out    

class MGPCConv_down(POTTS):
# the rest of the grid levels of the left branch
    def __init__(self,args=None, times=2, in_channels=1,out_channels=1,output_level=1):
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.output_level=output_level
        self.times=times
        self.kernel_size=np.minimum(self.kernel_size_max//(3**output_level),self.kernel_size_bound)
        self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=in_channels+3,out_channels=out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
        self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
        self.model.append(self.sigAct1)
        for i in np.arange(self.times-1):
            self.model.append(nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                      kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
            self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
            self.model.append(self.sigAct1)

    def forward(self,x,flist):

        out=self.pooling(x)

        out1=torch.cat((out,flist[self.output_level]),dim=1)
        if self.tau_explicit:
            out=self.out_channels* self.tau*self.model[0](out1)+torch.sum(out,dim=1,keepdim=True)/self.in_channels
        else:
            out=self.model[0](out1)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1):
            
            if self.tau_explicit:
                out=self.out_channels*self.tau*self.model[i*3+3](out)+torch.sum(out,dim=1,keepdim=True)/self.out_channels
            else:
                out=self.model[i*3+3](out)
            out=self.model[i*3+4](out)
            out=self.model[i*3+5](out)

        return out


class MGPCConv_up(POTTS):
# right branch of the V-cycle
    def __init__(self, args=None, times=2, in_channels=1,out_channels=1,output_level=1):
        super().__init__(args=args)

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.output_level=output_level
        self.times=times
        self.kernel_size=np.minimum(self.kernel_size_max//(3**output_level),self.kernel_size_bound)

        self.model=nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=in_channels+3,out_channels=out_channels,kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
        self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
        self.model.append(self.sigAct1)
        for i in np.arange(self.times-1):
            self.model.append(nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                      kernel_size=self.kernel_size,
                      stride=1,padding=self.kernel_size//2,bias=False))
            self.model.append(nn.BatchNorm2d(out_channels,affine=self.BNLearn))
            self.model.append(self.sigAct1)


    def forward(self,x,flist):

        out=F.interpolate(x, size=flist[self.output_level].shape[2:],  mode='nearest')

        out1=torch.cat((out,flist[self.output_level]),dim=1)
        
        if self.tau_explicit:
            out=self.out_channels*self.tau*self.model[0](out1)+torch.sum(out,dim=1,keepdim=True)/self.in_channels
        else:
            out=self.model[0](out1)
        out=self.model[1](out)
        out=self.model[2](out)

        for i in np.arange(self.times-1):

            if self.tau_explicit:
                out=self.out_channels*self.tau*self.model[i*3+3](out)+torch.sum(out,dim=1,keepdim=True)/self.out_channels
            else:
                out=self.model[i*3+3](out)
            out=self.model[i*3+4](out)
            out=self.model[i*3+5](out)

        return out

class Block(POTTS):
# construct blocks. Each block is a time step. 
    def __init__(self,args=None,in_channels=1):
        super(Block,self).__init__(args=args)
        self.sigAct=sigAct(args=args)

        self.downs=nn.ModuleList()
        self.downs.append(MGPCConv_first(args=args, times=self.times_list[0], in_channels=1, 
                        out_channels=self.mid_channels[0], kernel_size=self.kernel_size_max,))
        for i in np.arange(self.level_max-1):
            self.downs.append(MGPCConv_down(args=args, times=self.times_list[i+1], in_channels=self.mid_channels[i],
                                             out_channels=self.mid_channels[i+1],output_level=i+1,))
        self.ups=nn.ModuleList()    
        self.combineWeight=nn.Parameter(torch.randn(self.level_max-1,2), requires_grad=True)
        for i in np.flip(np.arange(self.level_max-1)):
            self.ups.append(MGPCConv_up(args=args, times=self.times_list[i], in_channels=self.mid_channels[i+1],
                                             out_channels=self.mid_channels[i],output_level=i,))
        if args.lambdaLearn:
            self.c=nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.c=torch.zeros(1).to(args.device)
            self.c.requires_grad=False
            
        self.final=nn.Conv2d(in_channels=self.mid_channels[0],out_channels=1,kernel_size=3, stride=1, padding=1,
                      bias=True)

    def forward(self,x,flist):
        out=x
        connect=[]
        for i in np.arange(self.level_max):

            out=self.downs[i](out,flist)
            if i<self.level_max-1:
                connect.append(out)

        connect=connect[::-1]

        for i in np.arange(self.level_max-1):

            out=self.ups[i](out,flist)
            if self.skip_connection:
                # connections between layers
                out=self.combineWeight[i,0]*out + self.combineWeight[i,1]*connect[i]


        if self.tau_explicit:
            out=self.tau*self.final(out)+torch.sum(out,dim=1,keepdim=True)/self.mid_channels[0]
        else:
            out=self.final(out)
        out=self.sigAct(out,self.c)

        return out


class POTTSNET(POTTS):
# assemble blocks
# times_list: number of layers for each grid level
# mid_channels: number of channels for each grid level
# num_blocks: number of time steps
# kernel_size_bound: the upper bound of kernel size. 
# tau: time step size
    def __init__(self, args=None, in_channels=1):
        super(POTTSNET,self).__init__(args=args)
        self.layer1=nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1,bias=False)
        self.BN1=nn.BatchNorm2d(1,affine=self.BNLearn)
        self.poolf=nn.MaxPool2d(kernel_size=2,stride=2)     
       
        self.timevarying=args.timevarying

        if self.timevarying:
            self.blocks=nn.ModuleList()
            for i in np.arange(self.num_blocks):
                self.blocks.append(Block(args=args))
        else:
            self.block=Block(args=args)

    def forward(self,f):

        # compute initial condition
        out=self.layer1(f)
        out=self.BN1(out)
        out=self.sig(out)

        # prepare f at different grid levels
        flist=[]
        flist.append(f)
        for i in range(self.level_max-1):
            f=self.poolf(f)
            flist.append(f)

        # start iterating
        if self.timevarying:
            for idx in range(self.num_blocks):
                out=self.blocks[idx](out,flist)
        else:
            for idx in range(self.num_blocks):
                out=self.block(out,flist)

        return out

import os
from PIL import Image

import torch
import torchvision.transforms as transforms


def get_gamma(LL,LH):
    _h = torch.log(LH).mean(dim=[-1,-2,-3],keepdim=True)
    _l = torch.log(LL).mean(dim=[-1,-2,-3],keepdim=True)
    gamma = _l/_h
    
    return gamma

def diffX(x):
    return torch.diff(x, dim=3, prepend=x[...,:1])
def diffY(x):
    return torch.diff(x, dim=2, prepend=x[...,:1,:])


def diffThreshX(x,a=1):
    nablaImg = torch.diff(x, dim=3, prepend=x[...,:1])
    mask = nablaImg.abs()>nablaImg.abs().mean()*a
    return mask*nablaImg

def diffThreshY(x,a=1):
    nablaImg = torch.diff(x, dim=2, prepend=x[...,:1,:])
    mask = nablaImg.abs()>nablaImg.abs().mean()*a
    return mask*nablaImg

def ease(x,eps=1e-6):
    return torch.clamp(x,min=eps,max=1-eps)

def loadImg(imgNo,isHigh=False):
    low_high = 'high' if isHigh else 'low'
    imgDir = '/home/liyujie/workspace/LOL/eval15/'+low_high+'/'
    imgPath = imgDir+str(imgNo)+'.png'
    img = Image.open(imgPath)
    return img

def loadImgVE(imgNo,isHigh=False):
    imgDir = '/home/liyujie/workspace/VE-LOL-L/VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Low_test/low00'
    if isHigh:
            imgDir = '/home/liyujie/workspace/VE-LOL-L/VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Normal_test/normal00'
    imgPath = imgDir+str(imgNo)+'.png'
    img = Image.open(imgPath)
    return img

def t2i(*args):
    tran = transforms.ToPILImage()
    if len(args) == 1:
        return tran(args[0].cpu().squeeze())
    l = []
    for t in args:
        t = t.cpu().squeeze()
        if t.dim() < 3:
            t = t.repeat(3, 1, 1)
        elif t.dim() > 3:
            t = t.squeeze()
        l.append(t)
    if len(l)>1:
        img = torch.cat(l,dim=2)
    return tran(img)

def i2t(i):
    i = i.convert("RGB")
    trans =transforms.ToTensor()
    return trans(i).unsqueeze(0)

def init_weights(m):
    # if type(m) == nn.Linear:
    if 'conv' in str(type(m)):
        m.weight.data.normal_(0.0, 1.0)
    if 'batchnorm' in str(type(m)):
        m.weight.data.normal_(0.0, 1.0)


def loadCkpt(model, optimizer, opt):
    startEpoch = 0
    if opt.continueTrain:
        try:
            ckptPath = os.path.join(opt.ckptDir, "Validation_LOWEST"+str(opt.continueEpoch)+".ckpt")
            if not os.path.exists(ckptPath):
                ckptPath = os.path.join(opt.ckptDir, "epoch_"+str(opt.continueEpoch)+".ckpt")
            checkpoint = torch.load(ckptPath,map_location=next(model.parameters()).device)
            if isinstance(model,torch.nn.DataParallel):
                # model.load_state_dict({key.replace('module.','').replace('l_conv.1','l_conv.0'):value for (key,value) in checkpoint['model_state_dict'].items()})
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except:
                    model.load_state_dict({'module.'+key:value for (key,value) in checkpoint['model_state_dict'].items()})
            else:
                model.load_state_dict({key.replace('module.',''):value for (key,value) in checkpoint['model_state_dict'].items()})
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            startEpoch = opt.continueEpoch
            # startEpoch = int(ckptPath.split("_")[-1].split(".")[0])
            print("Resuming from epoch ", startEpoch)
        except Exception as e:
            raise e
            print("Checkpoint not found! Initial the weights!")
            model.apply(init_weights)
    else:
        print("Initial the weights...")
        model.apply(init_weights)
        
    return startEpoch+1, model, optimizer,ckptPath

def saveCkpt(epoch, model, optimizer, opt):
    """ Saving model checkpoint """
    if opt.ckptDir == '':
        ckptDir = "./checkpoints_"+str(opt.iter)+str(opt.share)
    else:
        ckptDir = opt.ckptDir
        
    if not os.path.isdir(ckptDir):
        os.mkdir(ckptDir)
    checkpoint_path = os.path.join(ckptDir, "epoch_" + str(epoch) + ".ckpt")
    latest_path = os.path.join(ckptDir, "latest.ckpt")
    saveDataDic ={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
    torch.save(saveDataDic, checkpoint_path)
    torch.save(saveDataDic, latest_path)
    print("Saved checkpoint for epoch ", epoch)

def loadModuleCkpt(module, path):
    try:
        dic = torch.load(path,map_location=next(module.parameters()).device)['model_state_dict']
    except:
        dic = torch.load(path,map_location=next(module.parameters()).device)
    dic = {key.replace('module.',''):value for (key,value) in dic.items()}
    module.load_state_dict(dic,strict=False)
    
    

class AvgPool2dPadRep(torch.nn.Module):
    def __init__(self,kernel_size=3,stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool2d(kernel_size=kernel_size,stride=stride)
        
    def forward(self,x):
        y = torch.nn.functional.pad(x,tuple([self.kernel_size//2 for i in range(4)]),mode="replicate")
        return self.avg(y)
    
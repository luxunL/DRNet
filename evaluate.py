import warnings
warnings.filterwarnings('ignore')

import os

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image


import argparse
from model import *
from utils import *

import glob
LOL_dir = "/home/liyujie/workspace/LOL/eval15/"
VELOL_dir = "/home/liyujie/workspace/VE-LOL-L/VE-LOL-L-Cap-Full/test/"

import metrics
import lpips

ssim = metrics.ssim
psnr = metrics.psnr

class Inference(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        self.device = device
        self.model = DRNet3(32).to(device)
        loadModuleCkpt(self.model,ckpt_path)
        self.lpips = lpips.LPIPS(verbose=False).to(device)
        
    def open_img(self, file):
        return transforms.ToTensor()(Image.open(file)).unsqueeze(0).to(self.device)
        
    def forward(self,img0,imgH0, gamma=None, pad_size = 20):
        img = torch.nn.functional.pad(img0,pad=(0,pad_size,0,pad_size),mode="replicate")
        imgH = torch.nn.functional.pad(imgH0,pad=(0,pad_size,0,pad_size),mode="replicate")

        LL,RL = decomL(img)
        LH,RH = decomH(imgH)
        
        if gamma is None:
            gamma = get_gamma(LL,LH)
            
        LL = LL**(1/gamma)
        
        with torch.no_grad():
            L3,R3 = self.model(LL,RL)
        imgE = (L3*R3).clamp(0,1)

        return imgE[...,:-pad_size,:-pad_size]
    
    def test(self, imgs_dir = './testImgs/',gamma=2.2, save = False):
        # print(sum(p.numel() for p in self.model.parameters()))
        low_files = glob.glob(imgs_dir+"/*.png")

        output_dir = imgs_dir+"/output/model{gamma:.1f}/".replace('.','_')
        for file in low_files:
            file_name = os.path.basename(file)
            name = file_name.split('.')[0]
            img = transforms.ToTensor()(Image.open(file)).unsqueeze(0).to(self.device)
            _,_,h,w = img.shape
            imgE = self.forward(img, img**(1/gamma), gamma=gamma)
            imgE = imgE[...,:h,:w]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if save:
                save_image(imgE, output_dir+file_name)

        print(f'Enhanced images are saved in '+output_dir)
    
    def getLOL_pairs(self,dataset_dir = './LOL/'):
        low_dir = dataset_dir + "/eval15/low/"
        high_dir = dataset_dir + "/eval15/high/"
        
        low_files = glob.glob(low_dir+"/*.png")
        
        for file in low_files:
            file_name = os.path.basename(file)
            
            high_file = os.path.join(high_dir, file_name)
            img = self.open_img(file)
            imgH = self.open_img(high_file)
            
            yield (img, imgH, file_name)
    
    def getVELOL_pairs(self,dataset_dir = './VE-LOL-L/VE-LOL-L-Cap-Full/'):
        low_dir = dataset_dir + "VE-LOL-L-Cap-Low_test/"
        high_dir = dataset_dir + "VE-LOL-L-Cap-Normal_test/"
        
        low_files = glob.glob(low_dir+"/*.png")
        
        for file in low_files:
            file_name = os.path.basename(file)
            file_name = file_name.replace('low','')
            high_file = os.path.join(high_dir, 'normal'+file_name)
            img = self.open_img(file)
            imgH = self.open_img(high_file)
            
            yield (img, imgH, file_name)
    
    def testLOL(self,dataset_dir = './LOL/', save = False):
        # print(sum(p.numel() for p in self.model.parameters()))
        return self.evaluate(dataset_dir, self.getLOL_pairs, save)
            
    def testVELOL(self, dataset_dir = './VE-LOL-L/VE-LOL-L-Cap-Full/', save = False):
        # print(sum(p.numel() for p in self.model.parameters()))
        return self.evaluate(dataset_dir, self.getVELOL_pairs, save)
            
    def evaluate(self, dataset_dir, pair_generator, save = False, gamma = None):
        # print(sum(p.numel() for p in self.model.parameters()))
        if save:
            if gamma is not None:
                output_dir = os.path.join(dataset_dir, f"output/model{gamma:.1f}/".replace('.','_'))
            else:
                output_dir = os.path.join(dataset_dir, f"output/model/".replace('.','_'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        psnr_total = 0
        ssim_total = 0
        lpips_total = 0

        total_num = 0
        for img, imgH, file_name in pair_generator(dataset_dir):
            imgE = self.forward(img, imgH)
            if save:
                save_image(imgE, output_dir+file_name)

            psnr_total += psnr(imgE,imgH)
            ssim_total += ssim(imgE,imgH)
            lpips_total += self.lpips(imgE,imgH).squeeze()

            total_num += 1

        psnr_avg = psnr_total/total_num
        ssim_avg = ssim_total/total_num
        lpips_avg = lpips_total/total_num

        print(f':\tpsnr={psnr_avg:.4f}, ssim={ssim_avg:.4f}, lpips={lpips_avg:.4f}')
        if save:
            print(f'Enhanced images are saved in '+output_dir)

        return psnr_avg, ssim_avg, lpips_avg 
            

    

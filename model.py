import torch
import torch.nn as nn
from collections import OrderedDict
from utils import *

def decomH(I,u=0.1,beta1=0.5,beta2=0.1,k=3,g=1,T=5,back=True):
    eps = 1e-5
    L0 = I.max(dim=1,keepdim =True).values.clamp(eps,1)
    Sx = XH(L0)*(XH(L0).abs()>XH(L0).abs().mean())
    Sy = VX(L0)*(VX(L0).abs()>VX(L0).abs().mean())
    
    if back:
        L_backValue = torch.nn.AdaptiveMaxPool2d((7,7))(L0).mean(dim=[-1,-2],keepdim=True)
        L = (L_backValue+torch.e**(k*L0)*L0)/(1+torch.e**(k*L0))
    else:
        L = L0
    Q = L.clone()
    LambdaL = L*0.0
    VTV,HHT = getVH2(L)
    
    L_right = (1+u+g)*torch.eye(L.shape[-1]).to(L.device)+(beta1+beta2)*HHT
    L_right = torch.inverse(L_right)
    Q_left = g*torch.eye(L.shape[-2]).to(L.device)+(beta1+beta2)*VTV
    Q_left = torch.inverse(Q_left)
    
    for i in range(T):
        L_ = L.clone()
        
        I_R = L
        L_left = I_R+u*L0+beta1*XHT(Sx)+g*Q+LambdaL
        L = L_left@L_right
        L = L.clamp(eps,1)
        
        Q_right = beta1*VTX(Sy)+g*L-LambdaL
        Q = Q_left@Q_right
        Q = Q.clamp(eps,1)
        
        LambdaL += (Q-L)
        
        dL = (L_-L).abs().mean()/L.abs().mean()
        if dL<1e-3:
            # print(i,dL)
            break
    return Q,(I/Q).clamp(0,1)

def decomL(I):
    eps = 1e-5
    _,R = decomH(I)
    L = I.max(dim=1,keepdim =True).values.clamp(eps,1)
    return L, R


'''Basic Modules'''
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels*3
        self.MyConv=nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.MyConv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Conv(in_channels, out_channels),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e327a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
           
''' GEM ''' 
class RL_Block(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, RL):
        return self.convs(RL)

class GEM(nn.Module):
    def __init__(self,mid_channels = 32,out_channels=3):
        super().__init__()
        in_channels = 1+3
        
        self.LeakyReLU = nn.LeakyReLU()
        self.Tanh = nn.Tanh()
        
        self.NXB1 = nn.Conv2d( in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.NXB2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.NXB3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.NYB1 = nn.Conv2d( in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.NYB2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.NYB3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.RLB1 = RL_Block( in_channels,mid_channels,mid_channels)
        self.RLB2 = RL_Block(mid_channels,mid_channels,mid_channels)
        self.RLB3 = RL_Block(mid_channels,mid_channels,out_channels)
        
        
    def forward(self, R, L):
        RL0 = torch.cat([L,R],dim=1)
        NX0 = diffX(RL0)
        NY0 = diffY(RL0)
        
        RL1 = self.RLB1(RL0)
        NX1 = self.Tanh(self.NXB1(NX0)+diffX(RL1))
        NY1 = self.Tanh(self.NYB1(NY0)+diffY(RL1))
        
        RL2 = self.RLB2(self.LeakyReLU(RL1+NX1+NY1))
        NX2 = self.Tanh(self.NXB2(NX1)+diffX(RL2))
        NY2 = self.Tanh(self.NYB2(NY1)+diffY(RL2))
        
        RL3 = self.RLB3(self.LeakyReLU(RL2+NX2+NY2))
        NX3 = self.Tanh(self.NXB3(NX2)+diffX(RL3))
        NY3 = self.Tanh(self.NYB3(NY2)+diffY(RL3))
        
        Rout = self.LeakyReLU(RL3+NX3+NY3)
        
        return NX3,NY3,Rout

'''G/F Nets'''
class NetG(nn.Module):
    def __init__(self,mid_channels=32):
        super().__init__()
        mid_channels = mid_channels
        out_channels = 1
        map_num = 3
        
        self.down1 = Down(out_channels*map_num, mid_channels)
        self.inConv = nn.Sequential(
            Conv(mid_channels, mid_channels),
            Conv(mid_channels, mid_channels))
        self.up1 = Up(out_channels*map_num+mid_channels, mid_channels)
        self.neck = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1, bias=False)
        
    def forward(self,Lmath, Ls, Lin):
        x = torch.cat([Lmath, Ls, Lin],dim=1)
        x1 = self.inConv(self.down1(x))
        x2 = self.up1(x1,x)
        out = nn.LeakyReLU()(self.neck(x2)+Lin)
        return out
        
class NetF(nn.Module):
    def __init__(self,mid_channels=32):
        super().__init__()
        mid_channels = mid_channels
        out_channels = 3
        map_num = 3
        
        self.down1 = Down(out_channels*map_num, mid_channels)
        self.inConv = nn.Sequential(
            Conv(mid_channels, mid_channels),
            Conv(mid_channels, mid_channels))
        self.up1 = Up(out_channels*map_num+mid_channels, mid_channels)
        self.neck = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1, bias=False)
        
    def forward(self,Rmath, Rw, Rin):
        x   = torch.cat([Rmath, Rw, Rin],dim=1)
        x1 = self.inConv(self.down1(x))
        x2 = self.up1(x1,x)
        out = nn.LeakyReLU()(self.neck(x2)+Rin)
        return out

    
''' MPOM '''
def VTX(x):
    VTX = -torch.diff(x, dim=-2, append=x[...,:1,:])
    VTX[...,0,:] = -x[...,1,:]
    VTX[...,-1,:] = x[...,-1,:]
    return VTX

def VX(x):
    return torch.diff(x, dim=-2, prepend=x[...,:1,:])

def XHT(x):
    XHT = -torch.diff(x, dim=-1, append=x[...,:1])
    XHT[...,0] = -x[...,1]
    XHT[...,-1] = x[...,-1]
    return XHT

def XH(x):
    return torch.diff(x, dim=-1, prepend=x[...,:1])

def getVH(x):
    _,_,h,w = x.shape
    V = torch.eye(h)
    V[1:,:-1] -= torch.eye(h-1)
    V[0,0]=0
    H = torch.eye(w)
    H[:-1,1:] -= torch.eye(w-1)
    H[0,0]=0
    return V.to(x.device),H.to(x.device)

def getVH2(x):
    _,_,h,w = x.shape
    V = torch.eye(h)
    V[1:,:-1] -= torch.eye(h-1)
    V[0,0]=0
    H = torch.eye(w)
    H[:-1,1:] -= torch.eye(w-1)
    H[0,0]=0
    VTV = VTX(V.to(x.device))
    HHT = XHT(H.to(x.device))
    return VTV,HHT

    
def MPOM_L(L,Sx,Sy,beta1,beta2,g=1, T=3):
    Q = L.clone()
    Lambda = L*0.0
    VTV,HHT = getVH2(L)
    
    L_right = (1+g)*torch.eye(L.shape[-1]).to(L.device)+(beta1+beta2)*HHT
    L_right = torch.inverse(L_right)
    Q_left = g*torch.eye(L.shape[-2]).to(L.device)+(beta1+beta2)*VTV
    Q_left = torch.inverse(Q_left)
    
    for i in range(T):
        L_left = L+beta1*XHT(Sx)+g*Q+Lambda
        L = L_left@L_right
        L = L.clamp(0,1)
        
        Q_right = beta1*VTX(Sy)+g*L-Lambda
        Q = Q_left@Q_right
        Q = Q.clamp(0,1)
        
        Lambda += (Q-L)
    return Q

def MPOM_R(R,Wx,Wy,alpha,g=1,T=3):
    P = R.clone()
    Lambda = R*0.0
    VTV,HHT = getVH2(R)
    
    R_right = (1+g)*torch.eye(R.shape[-1]).to(R.device)+alpha*HHT
    R_right = torch.inverse(R_right)
    P_left = g*torch.eye(R.shape[-2]).to(R.device)+alpha*VTV
    P_left = torch.inverse(P_left)
    
    for i in range(T):
        R_left = R+alpha*XHT(Wx)+g*P+Lambda
        R = R_left@R_right
        R = R.clamp(0,1)
        
        P_right = alpha*VTX(Wy)+g*R-Lambda
        P = P_left@P_right
        P = P.clamp(0,1)
        
        Lambda += (P-R)
    return P
    
''' General Model'''
class LEM(nn.Module):
    def __init__(self, beta, mid_channels=32):
        super().__init__()
        self.NetG = NetG(mid_channels=mid_channels)
        self.beta1 = beta
        self.beta2 = 0.1

    def forward(self, L, R, SE):
        Sx,Sy,Ls = SE(L, R)
        # Lmath = LSMath(L,Sx,Sy,self.beta1,g=1,T=3)
        Lmath = MPOM_L(L,Sx,Sy,self.beta1,self.beta2,g=1,T=3)
        out  = self.NetG(Lmath, Ls, L)
        
        return out
    

class REM(nn.Module):
    def __init__(self, alpha, mid_channels=32):
        super().__init__()
        self.NetF = NetF(mid_channels=mid_channels)
        self.alpha = alpha

    def forward(self, R, L, WE):
        Wx,Wy,Rw = WE(R, L)
        Rmath = MPOM_R(R,Wx,Wy,self.alpha,g=1,T=3)
        out  = self.NetF(Rmath, Rw, R)
        
        return out

class DRNet3(nn.Module):
    def __init__(self,mid_channels=32,alpha0=100/4,beta0 = 100/4):
        super().__init__()
        self.WE1  = GEM(mid_channels,3)
        self.SE1  = GEM(mid_channels,1)
        self.LEM1 = LEM(1*beta0 ,mid_channels)
        self.REM1 = REM(1*alpha0,mid_channels)
        self.WE2  = GEM(mid_channels,3)
        self.SE2  = GEM(mid_channels,1)
        self.LEM2 = LEM(2*beta0 ,mid_channels)
        self.REM2 = REM(2*alpha0,mid_channels)
        self.WE3  = GEM(mid_channels,3)
        self.SE3  = GEM(mid_channels,1)
        self.LEM3 = LEM(4*beta0 ,mid_channels)
        self.REM3 = REM(4*alpha0,mid_channels)
        
    def forward(self, LL, RL):
        L1 = self.LEM1(LL, RL, self.SE1)
        R1 = self.REM1(RL, L1, self.WE1)
        L2 = self.LEM2(L1, R1, self.SE2)
        R2 = self.REM2(R1, L2, self.WE2)
        L3 = self.LEM3(L2, R2, self.SE3)
        R3 = self.REM3(R2, L3, self.WE3)
        
        return L3, R3
        
from evaluate import *

ckptDir = './ckpt/'
epoch = 150
cuda = True

save = False
file_name =  "epoch_"+str(epoch)+".ckpt"

ckpt_path = os.path.join(ckptDir, file_name)
if not os.path.exists(ckpt_path):
    print('There is no .ckpt file to load!')

device = 'cuda:0' if (cuda and torch.cuda.is_available()) else 'cpu'
model = Inference(ckpt_path = ckpt_path, device=device)

model.testLOL(save = True)
model.testVELOL(save = True)
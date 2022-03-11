# Modified version based on https://github.com/ZhengyuZhao/Targeted-Tansfer/eval_ensemble.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16_bn, mobilenet_v2
from models import resnet50, densenet121, wideresnet
from PIL import Image
from tqdm import tqdm
import numpy as np
import scipy.stats as st

device = torch.device("cuda:0")


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# define TI
def gkern(kernel_len=15, n_sig=3):
    x = np.linspace(-n_sig, n_sig, kernel_len)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    return kernel_raw / kernel_raw.sum()


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)


# define DI
def DI(x_in):
    rnd = np.random.randint(32, 41, size=1)[0]
    h_rem = 41 - rnd
    w_rem = 41 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        x_out = F.pad(F.interpolate(x_in, size=(rnd, rnd)),
                      (pad_left, pad_top, pad_right, pad_bottom), mode='constant',
                      value=0)
        return x_out
    else:
        return x_in


model_1 = resnet50()
checkpoint = torch.load('./models/trained/resnet50.pth.tar')
model_1.load_state_dict(checkpoint['state_dict'])

model_2 = wideresnet()
checkpoint = torch.load('./models/trained/wideresnet.pth.tar')
model_2.load_state_dict(checkpoint['state_dict'])

model_3 = densenet121()
checkpoint = torch.load('./models/trained/densenet121.pth.tar')
model_3.load_state_dict(checkpoint['state_dict'])

model_4 = vgg16_bn()
checkpoint = torch.load('./models/trained/vgg16.pth.tar')
model_4.load_state_dict(checkpoint['state_dict'])

model_5 = mobilenet_v2()
checkpoint = torch.load('./models/trained/mobilenetv2.pth.tar')
model_5.load_state_dict(checkpoint['state_dict'])


model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False
for param in model_5.parameters():
    param.requires_grad = False

model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)
model_5.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# values are standard normalization for CIFAR-10 images
norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
trn = transforms.Compose([transforms.ToTensor(), ])
images = np.load('test_data.npy')
labels = np.load('test_label.npy')
labels = np.argmax(labels, axis=1)

batch_size = 32
max_iterations = 300
num_batches = np.int32(np.ceil(len(images) / batch_size))
img_size = 32
lr = 2 / 255
epsilon = 8

adv_img = np.zeros_like(images)
for k in tqdm(range(0, num_batches)):
    batch_size_cur = min(batch_size, len(images) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.fromarray(images[k * batch_size + i]))
    labels = torch.tensor(labels[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')
    for t in tqdm(range(max_iterations)):
        logit1 = model_1(norm(DI(X_ori + delta)))
        logit2 = model_2(norm(DI(X_ori + delta)))
        logit3 = model_3(norm(DI(X_ori + delta)))
        logit4 = model_4(norm(DI(X_ori + delta)))
        logit5 = model_5(norm(DI(X_ori + delta)))
        logits = 0.24 * logit1 + 0.24 * logit2 + 0.24 * logit3 + 0.14 * logit4 + 0.14 * logit5
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        logit_dists = (-1 * real)
        loss = logit_dists.sum()
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
        grad_a = grad_c + 1 * grad_pre
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data + lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
    for j in range(batch_size_cur):
        x_np = transforms.ToPILImage()((X_ori + delta)[j].detach().cpu())
        adv_img[k * batch_size + j] = x_np

np.save('delta_linf.npy', adv_img.astype(np.int32) - images.astype(np.int32))
torch.cuda.empty_cache()

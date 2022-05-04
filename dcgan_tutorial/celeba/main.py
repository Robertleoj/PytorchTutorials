# Cell

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from glob import glob

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.backends import cudnn

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import math

torch.random.manual_seed(999)
random.seed(999)



cudnn.benchmark = True

# Cell
dataroot = '/media/king_rob/DataDrive/data/Celeba'
batch_size = 64

image_size = 128

n_channels = 3
n_latent = 100
n_feature_maps_g = 128
n_feature_maps_d = 64
beta1=0.5
ngpu = 1

data_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
])

class ImgFolder(Dataset):
    def __init__(self, path, data_transform):
        self.img_paths = glob(os.path.join(path, "*"))
        random.shuffle(self.img_paths)
        self.transform = data_transform
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        return data_transform(img)

    def __len__(self):
        return len(self.img_paths)

dataset = ImgFolder(dataroot, data_transform)

# Cell
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=10)


# Cell
device = torch.device('cuda')

def visualize(img_tensors, grid_size, title=None):
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    if title is not None:
        plt.title(title)

    grid = make_grid(img_tensors[:(grid_size ** 2)], padding=2, normalize=True, nrow=grid_size)
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)

# Cell
real_batch = next(iter(dataloader))
visualize(real_batch, 8, "Training images")

# Cell
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Cell
def get_convt_outdim(size, kernel_size, stride, padding):
    return (size - 1)* stride - 2 * padding + (kernel_size - 1) + 1




# Cell
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = n_feature_maps_g

        # find how many doublings we need to go to img_size
        curr = 4
        cnt = 0
        while curr < image_size:
            curr *= 2
            cnt += 1
        
        multiplier = 2 ** (cnt - 1)

        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(n_latent, ngf * multiplier, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * (multiplier)),
            nn.LeakyReLU(),
            nn.Conv2d(ngf * multiplier, ngf * multiplier, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * multiplier),
            nn.LeakyReLU()
        )

        upsample_layers = []

        while multiplier > 1:
            upsample_layers.extend([
                nn.ConvTranspose2d(ngf * multiplier, ngf * (multiplier // 2), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * (multiplier // 2)),
                # nn.ReLU(True),
                nn.LeakyReLU(),
                nn.Conv2d(ngf * (multiplier // 2), ngf * (multiplier // 2), kernel_size = 3, stride=1, padding=1),
                # nn.BatchNorm2d(ngf * (multiplier // 2)),
                nn.LeakyReLU()
            ])
            multiplier //= 2

        self.main = nn.Sequential(
            *upsample_layers
        )

        self.finalconvt = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, n_channels, kernel_size = 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.convt1(x)
        out = self.main(out)
        return self.finalconvt(out)

# Cell
netG = Generator().to(device)

netG.apply(weights_init)

# Cell
# netD = models.resnet18(pretrained=True)
# n_inputs = netD.fc.in_features
# netD.fc = nn.Linear(n_inputs, 1)



# netD.to(device)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = n_feature_maps_d

        # Find how many halvings we need

        main = []

        currsize = image_size

        main.extend([
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
        ])

        currsize //= 2

        multiplier = 1

        while currsize > 4:
            main.extend([
                nn.Conv2d(ndf * multiplier, ndf * (multiplier * 2), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * (multiplier * 2)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(ndf * multiplier * 2, ndf * multiplier * 2, kernel_size = 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * multiplier * 2),
                nn.LeakyReLU(0.1, inplace=True)
            ])

            currsize //= 2
            multiplier *= 2

        self.final = nn.Sequential(
            nn.Conv2d(ndf * multiplier, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.main = nn.Sequential(
            *main
        )


    def forward(self, x):
        features = self.main(x)
        result = self.final(features)
        return features, result

# Cell
netD = Discriminator().to(device)
netD.apply(weights_init)



# Cell
criterion = nn.BCELoss()

fixed_noise = torch.randn(25, n_latent, 1, 1, device=device)

real_label = 1.
fake_label = 0.

# lr_D = 0.00012
lr_D = 0.00012
# lr_G = 0.0002
lr_G = 0.00016

# optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999), weight_decay=0.1)
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999), weight_decay=0.05)
# optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999), weight_decay=0.1)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999), weight_decay=0.05)

# Cell
img_folder = '/home/king_rob/Desktop/Projects/torch_tutorials/dcgan_tutorial/celeba/training_pics/run2'

# Cell
# img_list = []
# G_losses = []
# D_losses = []
iters = 0
num_epochs = 5

for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):
        #######################
        # update D network
        ###############
        netD.zero_grad()

        # Train with all real
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)

        label_noise = (torch.randn((b_size, )) * 0.005).to(device)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) + label_noise
        label.clip_(0, 1)

        features_real, output = netD(real_cpu)
        output = output.view(-1)

        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        # train with all fake
        # run the fake batch through D again
        noise = torch.randn(b_size, n_latent, 1, 1, device=device)
        fake = netG(noise)

        label.fill_(fake_label)
        label = (label + label_noise).clip(0, 1)

        _, output = netD(fake.detach())
        output = output.view(-1)

        errD_fake = criterion(output, label)

        # Accumulate gradient
        errD_fake.backward()

        errD = errD_real + errD_fake
        # if errD > 1:
        optimizerD.step()

        D_G_z1 = output.mean().item()


        # update D

        grad_max_D = max([x.grad.norm() for x in  netD.parameters()])


        ###################
        # Update G net
        ##############
        # if errD.item() < 90:
        netG.zero_grad()
        label.fill_(real_label)

        features_fake, output = netD(fake)

        # errG = criterion(output, label)
        errG = criterion(output.view(-1), label)
        #  + 0.01 * (
        #     features_real.detach().view(b_size, -1) 
        #     - features_fake.view(b_size, -1)
        # ).mean(0).norm() ** 2

        errG.backward()

        D_G_z2 = output.mean().item()

        optimizerG.step()

        grad_max_G = max([x.grad.norm() for x in  netG.parameters()])
        # else :
            # errG, grad_max_G, D_G_z2 = None, None, None

        if i % 10 == 0:
            print(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}")
            print(f'\tLoss D: {errD.item():.4f}')
            if errG is not None:
                print(f'\tLoss G: {errG.item():.4f}')
            print(f'\tD(x): {D_x}')
            print(f'\tD(G(x)): {D_G_z1} / {D_G_z2}')
            print(f'\tmax(grad(G)): {grad_max_G} \tmax(grad(D)): {grad_max_D}')


        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        if (iters % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            grid = make_grid(fake, padding=2, normalize=True, nrow=5)
            # img_list.append(grid)

            plt.figure(figsize=(20, 20))
            plt.axis('off')
            plt.title(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}")
            grid = np.transpose(grid, (1, 2, 0))
            plt.imshow(grid)
            plt.savefig(os.path.join(img_folder, f"{epoch}_{num_epochs}_{i}_{len(dataloader)}.jpg"))
            plt.close()

        # for g in optimizerD.param_groups:
        #     g['lr'] = 0.00017

        # for g in optimizerG.param_groups:
        #     # g['lr'] = 0.0001
        #     g['lr'] *= 0.99


        iters += 1

# Cell
for g in optimizerD.param_groups:
    # g['lr'] = 0.00017
    g['weight_decay'] = 0

# Cell
for g in optimizerG.param_groups:
    # g['lr'] = 0.00016
    g['lr'] = 0.0001
    # g['weight_decay'] = 0

# Cell
model_path = "/media/king_rob/DataDrive/models/gans/dcgan_64_celeba/run2"
G_path = os.path.join(model_path, 'G.pt')
D_path = os.path.join(model_path, 'D.pt')

torch.save(netG.state_dict(), G_path)
torch.save(netD.state_dict(), D_path)



# Cell

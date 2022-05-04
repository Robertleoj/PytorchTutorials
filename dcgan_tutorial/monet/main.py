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

cudnn.benchmark = True

# Cell
dataroot = '/media/king_rob/DataDrive/data/PainterMyself/monet_jpg'
real_pic_root = '/media/king_rob/DataDrive/data/PainterMyself/photo_jpg'
batch_size = 8

image_size = 64

n_channels = 3
n_latent = 100
n_feature_maps_g = 128
n_feature_maps_d = 64
lr = 0.0002
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
        self.transform = data_transform
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        return data_transform(img)

    def __len__(self):
        return len(self.img_paths)

dataset = ImgFolder(dataroot, data_transform)
real_photos = ImgFolder(real_pic_root, data_transform)

# Cell
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=10)
photo_loader = DataLoader(dataset, batch_size=batch_size, num_workers=10)

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

        # input size is 3 x 256 x 256
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(n_channels, ngf * 2, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2),

        #     # shape = ngf * 2 * 128 * 128
        #     nn.Conv2d(ngf * 2, ngf * 4, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2),

        #     # shape = (ngf * 4) x 64 x 64
        #     nn.Conv2d(ngf * 4, ngf * 8, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2),

        #     # shape = (ngf * 8) x 32 x 32
        #     nn.Conv2d(ngf * 8, ngf * 16, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(ngf * 16),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2),

        #     # shape = (ngf * 16) x 16 x 16
        #     nn.Conv2d(ngf * 16, ngf * 32, kernel_size = 3, stride=1, padding=1),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(ngf * 32),
        # )

        self.downsample = nn.Sequential(
            
        )

        self.upsample = nn.Sequential(
            # input has shape ngf * 32 x 8 x 8
            nn.ConvTranspose2d(ngf * 32, ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),

            # State size - _ * 16 * 16
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),

            # State size - _ * 32 * 32
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),

            # State size - _ * 64 * 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2 , kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),


            # state size _ * 128 * 128
            nn.ConvTranspose2d(ngf * 2, n_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

            # state size _ * 256 * 256
        )

    def forward(self, x):
        out =  self.downsample(x)
        out = self.upsample(out)
        return out

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
        self.main = nn.Sequential(
            # input size = 256
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State size = _ x 128 x 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State size = _ x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State size = _ x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State size = _ x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State size =  _ x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size = _ x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size = _ x 1 x 1
        )

        self.fc = nn.Linear(ndf * 32, 1)

    def forward(self, x):
        out = self.main(x)
        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)
        return torch.sigmoid(out)

# Cell
netD = Discriminator().to(device)
netD.apply(weights_init)



# Cell
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, n_latent, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Cell
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 20

photo_iter = iter(photo_loader)
for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):
        #######################
        # update D network
        ###############
        netD.zero_grad()

        # Train with all real
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)

        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        # train with all fake
        # run the fake batch through D again
        photos = next(photo_iter, None)
        if photos is None:
            photo_iter = iter(photo_loader)
            photos = next(photo_iter)

        photos = photos.to(device)
        fake = netG(photos)

        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)

        errD_fake = criterion(output, label)

        # Accumulate gradient
        errD_fake.backward()

        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        # update D
        optimizerD.step()


        ###################
        # Update G net
        ##############

        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)

        errG = criterion(output, label)

        errG.backward()

        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}")
            print(f'\tLoss D: {errD.item():.4f}')
            print(f'\tLoss G: {errG.item():.4f}')
            print(f'\tD(x): {D_x}')
            print(f'\tD(G(x)): {D_G_z1} / {D_G_z2}')

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 40 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(photos).detach().cpu()
            grid = make_grid(fake, padding=2, normalize=True, nrow=int(batch_size ** (1/2)))
            img_list.append(grid)

            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.title(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}")
            grid = np.transpose(grid, (1, 2, 0))
            plt.imshow(grid)
            plt.savefig(f"{epoch}_{num_epochs}_{i}_{len(dataloader)}.jpg")
            plt.close()


        iters += 1




# Cell

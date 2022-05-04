import torch, torchvision
import os

# pretrained resnet18 model 

os.environ["TORCH_HOME"] = './models/renet'

# model won't download
model = torchvision.models.resnet18(pretrained=False)


# random data represents an image
data = torch.rand(1, 3, 64, 64)

# random label
labels = torch.rand(1, 1000)
# print(f'{labels=}')

# Run the data through the model - forward pass
prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step()






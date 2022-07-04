import torch
from torch import nn
from torchvision.utils import save_image


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# discriminator
D = nn.Sequential(
    nn.Linear(64*64, 1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(128, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 64*64),
    nn.Tanh())

G_path = "generator_iii.pkl"
G.load_state_dict(torch.load(G_path))
G.eval()
for i in range(100):
    z = to_cuda(torch.randn(1, 128))
    fake_images = G(z)  # Generate fake images
    fake_images = fake_images.view(fake_images.size(0), 1, 64, 64)
    fake_images = fake_images.repeat(1, 3, 1, 1)
    save_image(denorm(fake_images.data), f'iii/fake_{i}.png')

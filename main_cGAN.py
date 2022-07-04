import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
from torchvision import datasets
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image

"reference: https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/"

n_classes = 10
embedding_dim = 100
latent_dim = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                      nn.Linear(embedding_dim, 16))

        self.latent = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 512),
                      nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(nn.ConvTranspose2d(513, 64 * 8, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),
                      nn.Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)

        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_condition_disc = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                      nn.Linear(embedding_dim, 3 * 128 * 128))

        self.model = nn.Sequential(nn.Conv2d(2, 64, 4, 2, 1, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Flatten(),
                      nn.Dropout(0.4),
                      nn.Linear(4608, 1),
                      nn.Sigmoid()
                      )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 1, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(concat)
        return output



device = 'cuda' if torch.cuda.is_available() else 'cpu'
discriminator = Discriminator()
generator = Generator()

generator.load_state_dict(torch.load("generator_c.pkl"))
generator.eval()
classes_list = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
for j in range(20):
    for i in range(10):
        noise_vector = torch.randn(1, latent_dim, device=device)
        noise_vector = noise_vector.to(device)
        generated_image = generator((noise_vector, torch.LongTensor([i])))
        save_image(generated_image[0].data, f'/home/student/HW2/gan/CGAN2//img_{j}_{classes_list[i]}.png')


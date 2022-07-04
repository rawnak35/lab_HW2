import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
from torchvision import datasets
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image

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

        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
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
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(concat)
        return output


def generator_loss(label, fake_output):
    gen_loss = nn.BCELoss()(label, fake_output)
    return gen_loss


def discriminator_loss(label, output):
    disc_loss = nn.BCELoss()(label, output)
    return disc_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
discriminator = Discriminator()
generator = Generator()
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
G_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)


def train():
    train_dir = "/home/student/HW2/data_dup/train"
    data_transforms = transforms.Compose([ transforms.Grayscale(num_output_channels=3),transforms.Resize([128,128]),
                                           transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    print("The classes are: ", class_names)
    classes_list =  class_names
    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(len(train_dataloader))
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):

        D_loss_list, G_loss_list = [], []

        for index, (real_images, labels) in enumerate(train_dataloader):
            D_optimizer.zero_grad()
            real_images = real_images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).long()

            # train D
            real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
            fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))

            D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
            # print(discriminator(real_images))
            # D_real_loss.backward()

            noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
            noise_vector = noise_vector.to(device)

            generated_image = generator((noise_vector, labels))
            output = discriminator((generated_image.detach(), labels))
            D_fake_loss = discriminator_loss(output, fake_target)

            # train with fake
            # D_fake_loss.backward()

            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_loss_list.append(D_total_loss)

            D_total_loss.backward()
            D_optimizer.step()

            # Train generator with real labels
            G_optimizer.zero_grad()
            G_loss = generator_loss(discriminator((generated_image, labels)), real_target)
            G_loss_list.append(G_loss)

            G_loss.backward()
            G_optimizer.step()

        print(f" epoch: {epoch}, G_loss: {sum(G_loss_list)/len(G_loss_list)}, D_loss: {sum(D_loss_list)/ len(D_loss_list)}")
        save_image(generated_image.data[0], f'data_fake2/fake_images-{epoch}_l_{classes_list[labels.data[0].item()]}.png')
        save_image(real_images.data[0], f'data_real/images-{epoch}_l_{classes_list[labels.data[0].item()]}.png')

    torch.save(generator.state_dict(), './generator_c2.pkl')
    torch.save(discriminator.state_dict(), './discriminator_c2.pkl')

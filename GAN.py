import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Paths to your train and val directories
train_dir = "/home/student/HW2/data_train_GAN/train"


# Resize the samples and transform them into tensors
data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1),
                                      transforms.Resize([64, 64]), transforms.Normalize(mean=(0.5, ),
                                     std=(0.5, ))])

# Create a pytorch dataset from a directory of images


train_dataset = datasets.ImageFolder(train_dir, data_transforms)
class_names = train_dataset.classes
print("The classes are: ", class_names)

# Dataloaders initialization
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
print(len(train_dataloader))
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Discriminator
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

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(100):
    accuracy_real = 0
    accuracy_fake = 0

    for i, (images, orig_labels) in enumerate(train_dataloader):
        # Build mini-batch dataset
        batch_size = images.size(0)
        images = to_cuda(images.view(batch_size, -1))
        #print(images.shape)
        # Create the labels which are later used as input for the BCE loss
        real_labels = to_cuda(torch.ones(batch_size))
        fake_labels = to_cuda(torch.zeros(batch_size))

        # ============= Train the discriminator =============#
        # Compute BCE_Loss using real images where BCE_Loss(x, y):
        #         - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        D.train()
        G.train(False)  # <-> G.eval()

        outputs = D(images)  # Real images
        d_loss_real = criterion(outputs.squeeze(1), real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = to_cuda(torch.randn(batch_size, 128))
        fake_images = G(z)  # Generate fake images
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs.squeeze(1), fake_labels)
        fake_score = outputs

        # Backprop + Optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # =============== Train the generator ===============#
        # Compute loss with fake images
        D.train(False)
        G.train()
        z = to_cuda(torch.randn(batch_size, 128))
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs.squeeze(1), real_labels)

        # Backprop + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()


    print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
            'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                % (epoch, 200, i + 1, 600, d_loss.data, g_loss.data,
                real_score.data.mean(), fake_score.data.mean()))


    # Save real images
    if (epoch + 1) == 1:
        images = images.view(images.size(0), 1, 64, 64)
        save_image(denorm(images.data), './data_real/real_images.png')

    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 64, 64)

    save_image(denorm(fake_images.data[0]), 'data_fake_3/fake_images-%d.png' % (epoch + 1))

# Save the trained parameters
torch.save(G.state_dict(), './generator_iii.pkl')
torch.save(D.state_dict(), './discriminator_iii.pkl')

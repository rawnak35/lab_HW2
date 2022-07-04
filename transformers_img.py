import torch
from torchvision import models, transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes_list = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
train_dir = "/home/student/HW2/data_dup/train"


class AddGaussianNoise(object):
    """
    define transformer that adds Gaussian Noise to images
    """
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'




def crop(train_dir):
    """
    apply random crop to images add save
    :param train_dir: train data path
    :return: None
    """
    data_transforms = transforms.Compose([transforms.Resize([96, 96]),transforms.RandomCrop(64),transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    print("The classes are: ", class_names)
    classes_list = class_names
    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    file_path = '/home/student/HW2/random_crop/train'
    for index, (real_images, labels) in enumerate(train_dataloader):
        real_images = real_images[0]
        if index == 1:
            print(real_images.shape)
        label = labels[0].item()
        p = f"{file_path}/{classes_list[label]}/{index}.png"
        # print(p)
        save_image(real_images, p)


def noise(train_dir):
    """
    add  Gaussian Noise to images
    :param train_dir: train data path
    :return: None
    """

    data_transforms = transforms.Compose(
        [transforms.Resize([64, 64]), transforms.ToTensor(), AddGaussianNoise(0., 0.2)])

    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    print("The classes are: ", class_names)
    classes_list = class_names
    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    file_path = '/home/student/HW2/data_noise_2/train'
    for index, (real_images, labels) in enumerate(train_dataloader):
        real_images = real_images[0]
        if index == 1:
            print(real_images.shape)
        label = labels[0].item()
        p = f"{file_path}/{classes_list[label]}/{index}.png"
        # print(p)
        save_image(real_images, p)


def img_augmentation(train_dataloader, new_train_path):
    print("img_augmentation ")
    classes_dict = {0: "i", 1: "ii", 2: "iii", 3: "iv", 4: "v", 5: "vi", 6: "vii", 7: "viii", 8: "ix", 9: "x"}
    j = 0
    for i in range(len(train_dataloader)):

        orig_imgs, classes = next(iter(train_dataloader))
        classes_list = classes.tolist()

        for img, img_class in zip(orig_imgs,classes_list):
            print("j : ", j, " out of 2067")


            # RandomPerspective
            distortion_scale = random.randint(1, 100)/100
            perspective_transformer = transforms.RandomPerspective(distortion_scale=distortion_scale, p=0.3, fill=1)
            perspective_imgs = perspective_transformer(img)
            print("******perspective_imgs done********")


            #HorizontalFlip
            HorizontalFlip_transformer = transforms.RandomHorizontalFlip(p=0.3)
            HorizontalFlip_img = HorizontalFlip_transformer(perspective_imgs)
            print("******HorizontalFlip_img done********")

            #Rotation
            #RotationDegree = random.randint(1,360)
            Rotation_transformer = transforms.RandomRotation(degrees=(-180, 180))
            final_img = Rotation_transformer(HorizontalFlip_img)
            print("******Rotation_img done********")

            # GaussianBlur:
            GaussianBlur_prob = random.randint(1, 100) / 100
            if GaussianBlur_prob <= 0.3:
                blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                final_img = blurrer(final_img)
            print("******blurred_imgs done********")


            try:
                inp = final_img.numpy().transpose((1, 2, 0))
                plt.figure(figsize=(64, 64))
                #train_path = "/home/student/HW2/data_new/train/"
                path = new_train_path + classes_dict[img_class] + "/aug" + str(j) + ".png"
                plt.imsave(path, inp, format='png')
                plt.close('all')
                j += 1
            except:
                j += 1
                continue
    return
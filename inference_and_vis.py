import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
model_ft = models.resnet50(pretrained=False)
# Fit the last layer for our specific task
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_ft = model_ft.to(device)
model_path = "/home/student/HW2/dup_vi/trained_model.pt"
model_ft.load_state_dict(torch.load(model_path))

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 0.001

# Paths to your train and val directories
train_dir = "/home/student/HW2/data_dup_vi/train"
val_dir = "/home/student/HW2/data_orig/val"
# Resize the samples and transform them into tensors
data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

# Create a pytorch dataset from a directory of images
train_dataset = datasets.ImageFolder(train_dir, data_transforms)
val_dataset = datasets.ImageFolder(val_dir, data_transforms)
# Dataloaders initialization
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("train len : ", len(train_dataloader))
print("val len : ", len(val_dataloader))


def inference(model, dataloader, pahse):
    """
    get predicted labels by running model inference on dataloader data
    :param model: model
    :param dataloader: dataloader
    :param pahse: train or val - str
    :return:  None
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    running_corrects = 0
    running_loss = 0
    size = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        for y_true, y_pred in zip(labels.data, preds):
            size += 1
            y_true = y_true.item()
            y_true_list.append(y_true)
            y_pred = y_pred.item()
            y_pred_list.append(y_pred)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / size
    epoch_loss = running_loss  / size
    print(pahse, " acc :", epoch_acc.item())
    print(pahse, " loss :", epoch_loss)
    with open("y_true_list_dup_vi"+pahse+".pkl", "wb") as f:
        pickle.dump(y_true_list, f)
    with open("y_pred_list_dup_vi"+pahse+".pkl", "wb") as f:
        pickle.dump(y_pred_list, f)


inference(model_ft, train_dataloader, "train")
inference(model_ft, val_dataloader, "val")


def confusion_matrix(y_true, y_pred, phase, title):
    """
    plot confusion_matrix
    :param y_true: true labels - list
    :param y_pred: predicted labels - list
    :param phase: train or val - str
    :param title: title - method name
    :return:
    """
    classes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_categories = 10
    confusion = torch.zeros(10, 10)
    for i in range(len(y_true)):
        #confusion[classes_list[y_true[i]]][classes_list[y_pred[i]]] += 1
        confusion[y_true[i]][y_pred[i]] += 1
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticks(torch.arange(0, n_categories))
    ax.set_yticks(torch.arange(0, n_categories))

    # Set up axes
    ax.set_xticklabels(classes_list, rotation=90)
    ax.set_yticklabels(classes_list)
    plt.title(title)
    plt.savefig("/home/student/HW2/dup_vi/" + phase +"_confusion" )
    plt.clf()



with open("y_true_list_crop_vi_train.pkl", 'rb') as f:
    y_true_train = pickle.load(f)
with open("y_pred_list_crop_vi_train.pkl", 'rb') as f:
    y_pred_train = pickle.load(f)

with open("y_true_list_crop_vi_val.pkl", 'rb') as f:
    y_true_val = pickle.load(f)
with open("y_pred_list_crop_val.pkl", 'rb') as f:
    y_pred_val = pickle.load(f)


confusion_matrix(y_true_train, y_pred_train, "train","data_dup_vi confusion matrix -train-")
confusion_matrix(y_true_val, y_pred_val, "val", "data_crop_vi confusion matrix -val-")





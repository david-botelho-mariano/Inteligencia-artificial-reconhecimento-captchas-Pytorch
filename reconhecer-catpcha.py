import torch
from torchvision import datasets, models, transforms
import shutil
import matplotlib.pyplot as plt
import numpy as np

def visualize_model(model_treinado, id_imagem):
    shutil.copyfile(str(id_imagem) + ".jpg", "dataset/outros/" + str(id_imagem) + ".jpg")
    shutil.copyfile(str(id_imagem) + ".jpg", "dataset/fone-de-ouvido/" + str(id_imagem) + ".jpg")

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder("dataset/", data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True, num_workers=4)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes[0], image_datasets.classes[1]
    model_treinado.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_treinado(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                imshow(inputs.cpu().data[j])
                shutil.os.remove("dataset/fone-de-ouvido/" + str(id_imagem) + ".jpg")
                shutil.os.remove("dataset/outros/" + str(id_imagem) + ".jpg")
                return class_names[preds[j]]

def comecar_reconhecimento(model_treinado):
  for id_imagem in range(1,6,1):
    resultado = visualize_model(model_treinado, id_imagem)
    if resultado == "fone-de-ouvido":
        print("[ATENCAO] " + str(id_imagem) + ".jpg, eh: " + resultado)
    else:
        print("[INFO] " + str(id_imagem) + ".jpg, eh: " + resultado)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

device = torch.device("cpu")
model_treinado = torch.load('model-fones.key', map_location='cpu')

print("[INFO] inteligencia artificial treinada apenas para encontrar fone de ouvido em uma imagem")
comecar_reconhecimento(model_treinado)

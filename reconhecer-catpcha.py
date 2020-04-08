import torch
from torchvision import datasets, models, transforms
import shutil
import os

def ler_diretorio_imagens(path_captchas):
    lista_arquivos_desejados = []
 
    for lista_nome_diretorios, lista_nome_pastas, lista_nome_arquivos in os.walk(path_captchas):
        for arquivo in lista_nome_arquivos:
            lista_arquivos_desejados.append(os.path.join(lista_nome_diretorios, arquivo))  
    
    return lista_arquivos_desejados

def visualize_model(captcha_path, model_treinado):
    captcha_path_array = captcha_path.split("/")

    shutil.copyfile(captcha_path, "pytorch-config/outros/" + captcha_path_array[1])
    shutil.copyfile(captcha_path, "pytorch-config/fone-de-ouvido/" + captcha_path_array[1])

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder("pytorch-config/", data_transforms)
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
                shutil.os.remove("pytorch-config/fone-de-ouvido/" + captcha_path_array[1])
                shutil.os.remove("pytorch-config/outros/" + captcha_path_array[1])
                return class_names[preds[j]]

def comecar_reconhecimento(captcha_path, model_treinado):
    resultado = visualize_model(captcha_path, model_treinado)

    if resultado == "fone-de-ouvido":
        print("[ATENCAO] O arquivo: " + captcha_path + ", eh classificado como: " + resultado)
    else:
        print("[INFO] O arquivo: " + captcha_path + ", eh classificado como: " + resultado)

if __name__ == '__main__':
    print("[+] inteligencia artificial treinada apenas para encontrar fone de ouvido em uma imagem.")
    
    device = torch.device("cpu")
    model_treinado = torch.load('model-fones.key', map_location='cpu')
    lista_captchas_absolute_path = ler_diretorios_e_subdiretorios("captchas")

    for captcha_path in lista_captchas_absolute_path:
      comecar_reconhecimento(captcha_path, model_treinado)
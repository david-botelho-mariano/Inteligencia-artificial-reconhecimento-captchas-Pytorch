from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch
import os


def normalizar_imagem(arquivo):
    normalizacao = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    imagem = Image.open(arquivo).convert('RGB')
    return normalizacao(imagem).unsqueeze(0)


def prever(arquivo):
    tensor = normalizar_imagem(arquivo)
    outputs = model_treinado.forward(tensor)
    _, previsao = outputs.max(1)
    previsao_id = str(previsao.item())

    #imagem = Image.open(arquivo)
    #display(imagem)
    print([arquivo, imagem_classe[int(previsao_id)]])


def ler_diretorio_subdiretorio(pasta_dataset):
  lista_arquivos_desejados = []

  for lista_nome_diretorios, lista_nome_pastas, lista_nome_arquivos in os.walk(pasta_dataset):
    for arquivo in lista_nome_arquivos:
        lista_arquivos_desejados.append(os.path.join(lista_nome_diretorios, arquivo))

  for arquivo in lista_arquivos_desejados:
    prever(arquivo) 


if __name__ == '__main__':

  device = torch.device("cpu")
  model_treinado = torch.load('model-fones.key', map_location='cpu')

  imagem_classe = ['fone de ouvido', 'outros']

  ler_diretorio_subdiretorio("captchas-nunca-vistos/")

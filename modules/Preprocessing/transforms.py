# modules/Preprocessing/transforms.py

from torchvision import transforms

def get_image_transforms(image_size=224, is_train=False):
    """
    Cria e retorna um pipeline de transformações de imagem do torchvision.

    Args:
        image_size (int): O tamanho para o qual as imagens serão redimensionadas.
        is_train (bool): Se True, adiciona transformações de data augmentation
                         apropriadas para o treinamento.

    Returns:
        transforms.Compose: Um pipeline de transformações do torchvision.
    """
    # Estatísticas de normalização do ImageNet
    #Média e desvio padrão para os canais RGB
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        # Para treinamento, adiciona Data Augmentation para tornar o modelo mais robusto
        # Ex: Rotações aleatórias, inversões horizontais, etc.
        transform_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(), # Inverte a imagem horizontalmente com 50% de chance
            transforms.RandomRotation(10),     # Rotaciona a imagem em até 10 graus
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        # Para validação e teste, não usa data augmentation.
        # Apenas redimensiona, converte para tensor e normaliza.
        transform_pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        
    return transform_pipeline
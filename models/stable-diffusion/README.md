# U-NET + Autoencoder (Stable Diffusion)

Implementação da rede U-NET e Autoencoder para diffusion model (totalmente baseado na implementação do Stable Diffusion).

- Está incluido o modelo U-NET em `models/stable-diffusion/model/unet.py`;
- O Autoencoder está em `models/stable-diffusion/model/autoencoder.py`;
- O embedder utilizado é o CLIP e está em `models/stable-diffusion/model/clip_embedder.py`.
- A parte de atenção, tanto a cross-attention com o texto quanto o spatial transformer para self-attention da imagem, está em `models/stable-diffusion/model/attention.py`.
- Os samplers (que adicionam o ruído) estão em `models/stable-diffusion/model/samplers`.
- Vários scripts de ajuda estão em `models/stable-diffusion/scripts`;
    - Imagem para imagem, texto para imagem e in-painting.

## Como funciona
- O Autoencoder codifica a imagem de entrada em um espaço latente;
- O U-NET recebe a imagem, que pode ser um ruído gaussiano no espaço latente
    - Ele também recebe o texto codificado pelo CLIP através de um cross-attention
    - Ele usa o self-attention para capturar a relação espacial entre os pixels da imagem
    - Ele também recebe o timestep do ruído, que é usado para controlar a quantidade de ruído a ser adicionado através de um embedding posicional (senoidal) nas camadas ResNet;
    - Nem todas as camadas possuem atenção, as camadas com maior resolução costumam não ter cross-attention
    - Com a informação do timestep, o U-NET tenta prever o ruído que foi adicionado à imagem, e assim remover esse ruído
- O sampler (DDIM ou DDPM) adiciona o ruído à imagem de acordo com o timestep para treinar o modelo
    - O DDIM costuma ter menos passos
- O modelo é treinado para prever o ruído adicionado, e assim aprender a remover, dado um timestep específico que é passado para a ResNet
- O Autoencoder decodifica a imagem do espaço latente para o espaço de pixels, gerando a imagem final


## Execução de testes

Caso esteja usando uv, basta executar da seguinte forma:

```bash
uv sync
uv run [arquivo]
```
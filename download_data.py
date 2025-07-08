# download_data.py

import kagglehub
import os
import shutil
import glob

# 1. Defina o nosso diretório de destino final
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

print("Iniciando o download do dataset para o cache do Kaggle...")

try:
    # 2. Baixe o dataset sem especificar o 'path', e capture o caminho retornado
    downloaded_path = kagglehub.dataset_download(
        "raddar/tuberculosis-chest-xrays-shenzhen"
    )

    print(f"Download concluído em: {downloaded_path}")
    print(f"Movendo arquivos para o diretório de destino: {output_dir}")

    # 3. Move cada arquivo e pasta de dentro do diretório de download para o nosso diretório de destino
    for item in glob.glob(os.path.join(downloaded_path, '*')):
        shutil.move(item, output_dir)

    print("Arquivos movidos com sucesso!")

finally:
    # 4. (Opcional, mas boa prática) Limpa o diretório de cache agora que está vazio
    # para não deixar lixo no sistema de arquivos do container.
    if 'downloaded_path' in locals() and os.path.exists(downloaded_path):
        # A pasta pai também pode ser removida para uma limpeza completa.
        parent_dir = os.path.dirname(downloaded_path)
        shutil.rmtree(parent_dir)
        print(f"Diretório de cache limpo: {parent_dir}")
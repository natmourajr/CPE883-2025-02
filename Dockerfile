# Dockerfile

# Use uma imagem Python oficial como base
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código do projeto para o diretório de trabalho
COPY . .

# Expõe a porta (útil para futuras demos com APIs, como Flask/FastAPI)
EXPOSE 8888

# Comando padrão para manter o container rodando (útil para desenvolvimento)
CMD ["tail", "-f", "/dev/null"]

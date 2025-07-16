# CPE883-2025-02
Repositório para ser utilizado para a disciplina do programa de engenharia elétrica da Coppe CPE883 Tópicos Especiais em Aprendizado De Máquina

## Dockerfile Usage

To Build the Docker image and execute the container, run:

docker build -t cpe883-app .    # Create the image.
docker run --rm -it cpe883-app    # Run the container.

## Benchmark

To execute the benchmark using the W3 data, run:

git clone https://github.com/ricardovvargas/3w_dataset.git
cd 3w_dataset
sudo apt-get install p7zip-full

Install p7zip
- sudo apt-get install p7zip-full (ubuntu)
- brew install p7zip (mac)

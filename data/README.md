# Download and prepare datasets

# Images
## MNIST
- Execute _./download_mnist_
- Note that this file will download and prepare data in the following format: 
    - datasets/mnist_data/ 
        - images/ --> .jpg
            - test 
            - train
        - test_annotations.txt --> img_name | class_id
        - train_annotations.txt

## CIFAR
- Execute _./download_cifar_
- Note that this file will download and prepare data in the following format: 
    - datasets/cifar/ 
        - images/ --> .jpg
            - test 
            - train
        - test_annotations.txt --> img_name | class_id
        - train_annotations.txt

# Tabular
## Heart Disease
- Execute _./download_heart_
- Note that this file will download and prepare data in the following format: 
    - datasets/heart_disease/ 
        - test --> .csv
        - train --> .csv

## Sugestão do Natanael para datasets de desenvolvimento
    Temporal → i3w,
    Imagem → mnist
    Tabular → Natanael fazer dados tabulares do sonar
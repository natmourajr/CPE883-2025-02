# What is this package?
- Package containing all dataloaders that will be used in this project. 

# How to install?
- Make sure that you have the .venv activaded and updated with the _requirements.txt_ file. 
- Navigate to the root of the project and run the following command: 
- python -m pip install -e modules/Dataloaders

# Usage
- Example of using a MNIST dataloader
```python
from modules.Dataloaders.MNIST_dataloaders import mnist
mnist.MNISTDataset()
````

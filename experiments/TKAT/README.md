This project uses the KAN algorithm from:

- https://github.com/Adamdad/kat/tree/main

So, it's necessary to follow the installations:

# install torch and other things
pip install timm==1.0.3
pip install wandb # I personally use wandb for results visualizations
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e

## Setup the enviroment
pip install .[dev]    # Install requirements
python -m build    # Build the project (.tar.gz and .whl)

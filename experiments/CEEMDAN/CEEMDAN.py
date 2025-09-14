from CNN import CNN
from TKAN import TKAN
from DeepONet import DPNET
from CapsNet import Capsnet
from LSTM import LSTM
from GRU import GRU
from PatchTST import Path_Transformer




def main():
    """Execute each model the CEEMDAM problem"""

    # LSTM
    LSTM()

    # GRU
    GRU()

    # CNN
    CNN()

    # Training KAN
    TKAN()

    # DeepONet
    DPNET()

    # CapsNet
    Capsnet()

    # PatchTST
    Path_Transformer()
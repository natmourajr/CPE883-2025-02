@echo off
REM Script para rodar treino final do CapsNet com melhores hiperparâmetros (Windows)

REM Ative seu ambiente virtual se necessário:
REM call C:\Users\betom\activate.bat meu_env

python cifar_capsnet_2.py ^
  --data_dir "G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\Antonio_Alberto\data\cifar10" ^
  --save_dir "G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\Antonio_Alberto\experiments\CIFARCapsNet\result" ^
  --epochs 50 ^
  --batch_size 128 ^
  --k_folds_final 3 ^
  --optimizer_name SGD ^
  --lr 0.01

pause

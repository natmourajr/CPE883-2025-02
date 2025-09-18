@echo off
setlocal 
REM Script para rodar busca de hiperparâmetros com Optuna no CapsNet (Windows)

REM Ative seu ambiente virtual se necessário:
REM call C:\Users\betom\activate.bat capsnet
REM python -m pip install --upgrade pip
REM python -m pip install scikit-learn tqdm pillow matplotlib optuna

set "PROJECT_ROOT=G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\antonio_alberto\experiments\CIFARCapsNet"

REM === Entrar no diretório do projeto ===
REM cd /d "%PROJECT_ROOT%"

REM === Garantir que Python importe os pacotes do projeto ===
REM set "PYTHONPATH=%G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\antonio_alberto\experiments\CIFARCapsNet\run_optuna.bat%;%PYTHONPATH%"
python "%PROJECT_ROOT%\cifar_capsnet_4.py" ^
  --data_dir "G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\antonio_alberto\data\cifar10" ^
  --save_dir "G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\antonio_alberto\experiments\CIFARCapsNet\result3" ^
  --epochs 50 ^
  --epochs_optuna 25 ^
  --batch_size 16 ^
  --batch_size_optuna 16 ^
  --use_optuna ^
  --n_trials 20 ^
  --k_folds_optuna 5 
  --final_no_kfold ^
  --final_val_split 0.1

endlocal

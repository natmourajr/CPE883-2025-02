
#!/bin/bash

# Default values
EPOCHS=100
BATCH_SIZE=100
LR=0.005
WEIGHTS="/home/eduardo/doc/CPE883-2025-02/results_cifar100_20/vit/fold_4.pkl"
FOLDS=5
DATA_DIR="/home/eduardo/doc/CPE883-2025-02/datasets/cifar100/"
MODEL="vit"
TEST=1
CIFAR_TYPE="100"
CLASS_TYPE="superclass"  # or "superclass"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--epochs)
			EPOCHS="$2"
			shift 2
			;;
		--batch_size)
			BATCH_SIZE="$2"
			shift 2
			;;
		--lr)
			LR="$2"
			shift 2
			;;
		-w|--weights)
			WEIGHTS="$2"
			shift 2
			;;
		-f|--folds)
			FOLDS="$2"
			shift 2
			;;
		--data_dir)
			DATA_DIR="$2"
			shift 2
			;;
		--test)
			TEST=1
			shift
			;;
		--cifar_type)
			CIFAR_TYPE="$2"
			shift 2
			;;
		--model)
			MODEL="$2"
			shift 2
			;;
		--class_type)
			CLASS_TYPE="$2"
			shift 2
			;;
		*)
			shift
			;;
	esac
done

CMD="python3 generate_confusion_matrix.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --folds $FOLDS --data_dir $DATA_DIR --model $MODEL --cifar_type $CIFAR_TYPE --test $TEST --class_type $CLASS_TYPE"
if [[ -n "$WEIGHTS" ]]; then
	CMD+=" --weights $WEIGHTS"
fi

echo "Running: $CMD"
eval $CMD

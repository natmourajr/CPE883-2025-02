
#!/bin/bash

# Default values
EPOCHS=50
BATCH_SIZE=100
LR=0.005
WEIGHTS=""
FOLDS=5
DATA_DIR="/home/eduardo/doc/CPE883-2025-02/datasets/cifar10/"

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
		*)
			shift
			;;
	esac
done

CMD="python3 resnet50.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --folds $FOLDS --data_dir $DATA_DIR"
if [[ -n "$WEIGHTS" ]]; then
	CMD+=" --weights $WEIGHTS"
fi

echo "Running: $CMD"
eval $CMD

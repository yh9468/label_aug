CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.007 --workers 4 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --augmented-bg 4 --dataset pascal

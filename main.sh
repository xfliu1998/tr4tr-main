# train or predict
experiment_mode='train'
echo ${experiment_mode}

CUDA_VISIBLE_DEVICES=0,1 python main.py --nproc_per_node=2 \
                                        --mode="${experiment_mode}"

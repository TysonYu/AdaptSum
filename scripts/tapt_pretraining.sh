python ./src/tapt_pretraining.py -path=./dataset/debate/TAPT-data/train.source \
                                -dm=debate \
                                -visible_gpu=0 \
                                -save_interval=100 \
                                # -recadam \
                                # -logging_Euclid_dist
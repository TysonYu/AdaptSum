python ./src/dapt_pretraining.py -path=./dataset/debate/debateorg.txt \
                                -dm=debate \
                                -visible_gpu=0 \
                                -save_interval=10000 \
                                # -recadam \
                                # -logging_Euclid_dist
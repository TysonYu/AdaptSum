python ./src/sdpt_pretraining.py -data_name=SDPT-cnn_dm \
                                -visible_gpu=0 \
                                -saving_path=SDPT_save \
                                -start_to_save_iter=0 \
                                -save_interval=10000 \
                                # -recadam \
                                # -logging_Euclid_dist \
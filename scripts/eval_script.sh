#!/bin/bash
python eval.py --data_dir ../data/audio_and_txt_files/ --folds_file ../data/patient_list_foldwise.txt --batch_size 64 --num_worker 4 --test_fold 4 --checkpoint models/ckpt_best.pkl --steth_id -1

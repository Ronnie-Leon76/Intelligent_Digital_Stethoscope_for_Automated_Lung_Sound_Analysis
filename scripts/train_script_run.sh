#!/bin/bash
python train.py --data_dir ../data/audio_and_txt_files/ --folds_file ../data/patient_list_foldwise.txt --model_path models_out --lr 1e-3 --batch_size 64 --num_worker 4 --start_epochs 0 --epochs 200 --test_fold 4

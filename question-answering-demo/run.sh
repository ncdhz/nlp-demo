date=`date -d "0 day ago" +"%Y%m%d-%H%M"`

nohup accelerate launch run.py --train_file /path/to/train --test_file /path/to/test --base_model_name bert-base-uncased --do_train --train_batch_size 8 --test_batch_size 8 --epochs 3 --learning_rate 2e-5 > $date.log 2>&1 &
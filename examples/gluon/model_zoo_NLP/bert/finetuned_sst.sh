pip install -U regex
python3 finetune_classifier.py --task_name SST --batch_size 32 --gpu 0 --log_interval 100  \
       --bert_model roberta_12_768_12 --epochs 10 --dtype float32 --early_stop 2 --lr 2e-5 \
       --bert_dataset openwebtext_ccnews_stories_books_cased --warmup 0.06 --no_best_model
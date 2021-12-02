pip install -U regex
python finetune_classifier.py --task_name MNLI --batch_size 32 --gpu 0 --log_interval 100  \
       --bert_model roberta_12_768_12 --epochs 3 --dtype float32 --early_stop 2 --lr 1e-5  \
       --bert_dataset openwebtext_ccnews_stories_books_cased --warmup 0.06 --no_best_model

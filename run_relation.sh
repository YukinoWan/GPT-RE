python3 gpt3_relation.py \
    --task semeval \
    --model text-davinci-003 \
    --num_test 100 \
    --example_dataset "./dataset/semeval_gpt/train.json" \
    --test_dataset "./dataset/semeval_gpt/sub_test_1.json" \
    --fixed_example 0\
    --fixed_test 1\
    --num_per_rel 0 \
    --num_na 0 \
    --no_na 0\
    --num_run 1 \
    --seed 0 \
    --random_label 0 \
    --reasoning 0 \
    --use_knn 1 \
    --k 27


## Introduction
### This the repository for the paper "GPT-RE: In-context Learning for Relation Extraction using Large Language Models"
This repository now supports GPT-Random, GPT-SimCSE and the reasoning logic in the paper, GPT-RE_FT method is not supported in the current version, we will soon update.


## Usage  
```   
bash run_relation_ace.sh   
```   
In the example script file `run_relation_ace.sh`, there are multiple options:   
`--task`: name of the task, eg. semeval    
`--model`: name of the model used in gpt3, eg. text-davinci-003   
`--num_test`: number of examples in the test subset, eg. 100, this should be smaller than the length of the test dataset    
`--example_dataset`: the file to choose demonstrations   
`--test_dataset`: the path to the test data, note that it should be changed to "test.json" if you want to work on the whole test data   
`--fixed_example`: if 1, then the demonstration examples will be fixed during the test; if 0, then for each test example, the demonstrations will be retrieved again. In kNN setup, it should be 0   
`--fixed_test`: if 1, then the test dataset will be fixed (this option can be ignored as you can directly change test data by `--test_dataset`   
`--num_per_rel`: the number of examples per relation type chosed to be the demonstrations, this should be 0 in kNN setup   
`--num_na`: the number of NA examples chosed to be the demonstrations, this should be 0 in both w/o NA setup and kNN setup   
`--num_run`: keep 1   
`--seed`: random seed   
`--random_label`: if 1, the model will change the gold lables to random labels in the demonstration   
`--reasoning`: if 1, the model will add reasoning to the demonstrations   
`--use_knn`: if 1, use kNN in demonstrations   
`--k`: top k in kNN   
`--var`: you can igore it, keep 0
`--reverse`: reverse the order of demonstrations, if 0, the default order is that more similar demonstrations will be placed at the top.
`--verbalize`: you can ignore it, keep 0
`--entity_info`: entity-aware sentence similarity in our paper
`--structure`: a trial to use structured prompt, you can ignore it
`--use_ft`: the fine-tuned representation, this option is not yet supported in the current version
`--self_error`: you can ignore it ,keep 0
`--use_dev, store_error_reason, discriminator`: keep 0




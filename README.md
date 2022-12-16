## Usage  
```   
bash run_relation.sh   
```   
In the file `run_relation.sh`, there are multiple options:   
`--task`: name of the task, eg. semeval    
`--model`: name of the model used in gpt3, eg. text-davinci-003   
`--num_test`: number of the test examples, eg. 100, this should be smaller than the length of the test dataset    
`--example_dataset`: the file to choose demostrations   
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




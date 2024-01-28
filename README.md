For the usage of faiss, I recommend to use conda environment (python 3.10).

# Installtion
`conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl`
`pip install torch`
`pip install transformers`


# Example
I provide a toy example in `knn.py` for the usage of faiss in the function `single_search`.

Input:
`xq`: a query embedding, e.g., "hello"
`xb`: a list of candidate embeddings, e.g., ["have a nice day", "你好", "おはようございます", "where are you from?", "what is this?"]
`k`: top-k similar candidates

Output:
`D`: the distance between `xq` and the top-l candidates for each query
`I`: the list of index of top-k candidates for each query, e.g., [[1 2 0]]







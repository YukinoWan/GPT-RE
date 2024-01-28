import numpy as np
import faiss
from tqdm import tqdm
import json
import sys
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel




def single_search(xq, xb ,k):
    d = len(xb[0])
    cpu_index = faiss.IndexFlatL2(d)

    gpu_index = faiss.index_cpu_to_all_gpus(
        cpu_index
    )
    gpu_index.add(xb)

    D, I = gpu_index.search(xq, k)
    return D, I

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def e5_encoder(input_str, model, tokenizer):
    batch_dict = tokenizer(input_str, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1).detach().numpy()

def encode(query, candidate_list, model_name_or_path):
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    query_e5 = e5_encoder(query, model, tokenizer)
    
    e5_embed_list = []
    print("start encoding candidates ...")
    for _item in tqdm(candidate_list):
        embeddings = e5_encoder(_item, model, tokenizer)
        e5_embed_list.append(embeddings[0])

    return query_e5, np.array(e5_embed_list)


if __name__ == "__main__":
    candidate_list = ["have a nice day", "你好", "おはようございます", "where are you from?", "what is this?"]
    query = "hello"
    top_k = 3

    query_e5, candidate_e5_list = encode(query, candidate_list, "intfloat/multilingual-e5-large")

    D,I = single_search(query_e5, candidate_e5_list, top_k)

    print(D)
    print(I)
    for idx in I[0]:
        print(candidate_list[idx])



from simcse import SimCSE
from shared.prompt import instance
from transformers import pipeline
import numpy as np
#model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


#embeddings = model.encode("A woman is reading.")

#sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
#sentences_b = ['He plays guitar.', 'A woman is making a photo.']
#similarities = model.similarity(sentences_a, sentences_b)
#print(similarities)


#sentences = ['A woman is reading.', 'A man is playing a guitar.']
#model.build_index(sentences)
#results = model.search("He plays guitar.")

#print(results)


def find_knn_example(model, test_dict, train_dict, k, entity_info):
    if entity_info:
        test_sentences = instance(test_dict).reference
    else:
        test_sentences = " ".join(test_dict["sentences"][0])
    test_id = test_dict["doc_key"]
    label_other = 0
    #train_dict = {" ".join(x["sentences"][0]):x for x in train_list}
    #train_sentences = [x for x in train_dict.keys()]
    
    #print(len(test_sentences))
    #print(len(train_sentences))
    #model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    #model.build_index(train_sentences, device="cpu")

    #for x in test_sentences:

    #    knn_result = model.search(x, device="cpu", threshold=0.3, top_k=3)
    #    print(knn_result)
    #    assert False
    knn_result = model.search(test_sentences, device="cpu", threshold=0.0, top_k=k)
    #print(knn_result)
    knn_list = [train_dict[x[0]] for x in knn_result]
    #if var and not no_na:
    #    label_other = knn_variance(knn_list)

    #print(train_sentences[0])
    #print(knn_list)
    #assert False
    return knn_list

def find_lmknn_example(gpu_index_flat, test_dict, train_dict, train_sentences, k):
    
    test_sentence = instance(test_dict).lm_mask
    extractor = pipeline(model="roberta-large", task="feature-extraction")
    result = extractor(test_sentence, return_tensors=True)
    
    embed = result.detach().numpy().copy()
    xq = np.array([embed[0][-3]])

    print(xq.shape)
    D, I = gpu_index_flat.search(xq, k)
    print(I)

    knn_list = [train_dict[train_sentences[i]] for i in I[0,:k]]

    return knn_list

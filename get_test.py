from gpt3_relation import get_test_example
import random
import json
import sys
from shared.const import semeval_reltoid


seed = sys.argv[1]
random.seed(seed)
with open("./dataset/semeval_gpt/sub_test_{}.json".format(seed), "w") as f:
    test_dict = get_test_example("./dataset/semeval_gpt/test.json", semeval_reltoid)
    flat_examples = [item for sublist in test_dict.values() for item in sublist]
    test_examples = random.sample(flat_examples, 100)
    for example in test_examples:
        json.dump(example, f)
        f.write("\n")

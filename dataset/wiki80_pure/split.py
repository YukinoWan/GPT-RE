import json
from random import random
from random import seed

seed(0)

def split(infile):
    with open(infile, "r") as f:
        #whole_list = []
        train_list = []
        dev_list = []
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            #whole_list.append(tmp_dict)
            tmp_value = random()
            if tmp_value > 0.95:
                train_list.append(tmp_dict)
            else:
                dev_list.append(tmp_dict)
        return train_list, dev_list


if __name__ == "__main__":
    train_list, dev_list = split("train.json")
    with open("0.05train.json", "w") as f:
        for sample in train_list:
            json.dump(sample, f)
            f.write("\n")

    #with open("0.99ent_pred_dev.json", "w") as f:
    #    for sample in dev_list:
    #        json.dump(sample, f)
    #        f.write("\n")

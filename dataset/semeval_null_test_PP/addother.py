import json
import random

random.seed(1)


Predefined_rel = ["Cause-Effect", "Component-Whole", "Entity-Destination"]
Undefined_rel = ["Product-Producer", "Entity-Origin","Member-Collection", "Message-Topic", "Content-Container", "Instrument-Agency"]

start = 0
end = 1
num = 600

pre_num = 200
un_num = num / (end - start)
#print(un_num)

rel_dict = dict()
with open("./train_1.json", "w") as outfile:
    with open("../semeval_gpt/train.json", "r") as infile:
        for line in infile.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                continue
            if tmp_dict["relations"][0][0][4] in rel_dict.keys():
                rel_dict[tmp_dict["relations"][0][0][4]].append(tmp_dict)
            else:
                rel_dict[tmp_dict["relations"][0][0][4]] = [tmp_dict]

    for rel_list in rel_dict.values():
        random.shuffle(rel_list)

    for rel in rel_dict.keys():
        if rel in Predefined_rel:
            for tmp_dict in rel_dict[rel][:pre_num]:
                json.dump(tmp_dict, outfile)
                outfile.write("\n")
        elif rel in Undefined_rel[start:end]:
            for tmp_dict in rel_dict[rel][:int(un_num)]:
                tmp_dict["relations"] = [[[tmp_dict["ner"][0][0][0], tmp_dict["ner"][0][0][1], tmp_dict["ner"][0][1][0], tmp_dict["ner"][0][1][1], "Other"]]]
                json.dump(tmp_dict, outfile)
                outfile.write("\n")

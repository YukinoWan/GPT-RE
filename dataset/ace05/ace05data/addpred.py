import json

with open("ent_pred_test.json", "w") as outfile:
    with open("train.json", "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            tmp_dict["predicted_ner"] = tmp_dict["ner"]

            json.dump(tmp_dict, outfile)
            outfile.write("\n")


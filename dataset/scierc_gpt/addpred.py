import json

with open("./test.json", "w") as outfile:
    with open("./pre_test.json", "r") as f:
        _id = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            tmp_dict["predicted_ner"] = tmp_dict["ner"]
            tmp_dict["doc_key"] = str(_id) + "oftrain"
            _id += 1

            json.dump(tmp_dict, outfile)
            outfile.write("\n")


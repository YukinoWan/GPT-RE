import json

with open("train_50.json", "w") as outfile:
    with open("train_50_ori.json", "r") as infile:
        for line in infile.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                tmp_dict["relations"] = [[[tmp_dict["ner"][0][0][0], tmp_dict["ner"][0][0][1], tmp_dict["ner"][0][1][0], tmp_dict["ner"][0][1][1], "Other"]]]
            json.dump(tmp_dict, outfile)
            outfile.write("\n")

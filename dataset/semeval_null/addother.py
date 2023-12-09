import json

with open("train.json", "w") as outfile:
    with open("../semeval_gpt/train.json", "r") as infile:
        for line in infile.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"][0][0][4] == "Other":
                json.dump(tmp_dict, outfile)
                outfile.write("\n")
            else:
                continue

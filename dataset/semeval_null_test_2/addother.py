import json

with open("./train_03_notarget.json", "w") as outfile:
    with open("../semeval_gpt/train.json", "r") as infile:
        for line in infile.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                continue
            elif tmp_dict["relations"][0][0][4] == "Content-Container" or tmp_dict["relations"][0][0][4] == "Message-Topic" or tmp_dict["relations"][0][0][4] == "Instrument-Agency":
                tmp_dict["relations"] = [[[tmp_dict["ner"][0][0][0], tmp_dict["ner"][0][0][1], tmp_dict["ner"][0][1][0], tmp_dict["ner"][0][1][1], "Other"]]]   
            elif tmp_dict["relations"][0][0][4] == "Member-Collection" or tmp_dict["relations"][0][0][4] == "Product-Producer" :
                continue
            json.dump(tmp_dict, outfile)
            outfile.write("\n")

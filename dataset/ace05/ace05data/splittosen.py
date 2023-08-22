import json

with open("../dev.json","w") as outfile:
    with open("./dev.json","r") as f:
        for line in f.read().splitlines():
            print("n")
            pre_len = 0
            tmp_dict = json.loads(line)
            for _id in range(len(tmp_dict["sentences"])):
                if len(tmp_dict["ner"][_id]) < 2:
                    pre_len += len(tmp_dict["sentences"][_id])
                    continue
                else:
                    #assert False
                    new_sentence = tmp_dict["sentences"][_id]
                    ners = tmp_dict["ner"][_id]
                    relations = tmp_dict["relations"][_id]
                    print("new--------------")
                    print(ners)
                    print(relations)

                    relation_dict = {}
                    #if relations != []:
                    #    for rel in relations:
                    #        relation_dict[rel[0:4]] = rel[4]

                    for head in ners:
                        for tail in ners:
                            if head == tail:
                                print("skip")
                                continue
                            else:
                                new_ner = [[head[0]-pre_len,head[1]-pre_len,head[2]], [tail[0]-pre_len,tail[1]-pre_len,tail[2]]]
                                if relations == []:
                                    print("go this way")
                                    new_relation = []
                                else:
                                    for rel in relations:
                                        if [head[0],head[1],tail[0],tail[1]] == rel[0:4]:
                                            print("-----")
                                            print(rel)
                                            print([head[0],head[1],tail[0],tail[1]] )
                                            
                                            print(pre_len)
                                            new_relation = [[rel[0]-pre_len,rel[1]-pre_len,rel[2]-pre_len,rel[3]-pre_len,rel[4]]]
                                            print(new_relation)
                                            print(new_ner)
                                            break
                                            
                                        else:
                                            new_relation = []
                                new_dict = {}
                                new_dict["sentences"] = [new_sentence]
                                new_dict["ner"] = [new_ner]
                                new_dict["relations"] = [new_relation]
                                print(new_dict)
                                print(pre_len)
                                
                                if new_dict["relations"] != [[]] and new_dict["ner"][0][0][0] != new_dict["relations"][0][0][0]:
                                    print(new_dict)
                                    print(pre_len)
                                    print(new_ner)
                                    print(new_sentence[new_ner[0][0]:(new_ner[0][1]+1)])
                                    print(tmp_dict)
                                    assert False

                                json.dump(new_dict, outfile)
                                outfile.write("\n")

                    pre_len += len(tmp_dict["sentences"][_id])
                                


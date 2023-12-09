import json
import sys
import random
import itertools


def split(infile, val_ratio, test_ratio):
    with open(infile, "r") as f:
        instance_dict = dict()
        train_list = list()
        val_list = list()
        test_list = list()
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            #instance_list.append(str(tmp_dict))
            if tmp_dict["relations"] == [[]]:
                relation_type = "None"
            else:
                relation_type = tmp_dict["relations"][0][0][4]
            if relation_type in instance_dict.keys():
                instance_dict[relation_type].append(tmp_dict)
            else:
                instance_dict[relation_type] = [tmp_dict]
        print("num of relation types is : %d"%(len(instance_dict.keys())))

        for rel in instance_dict.keys():
            tmp_list = instance_dict[rel]
            #print(len(tmp_list))
            random.shuffle(tmp_list)
            if val_ratio * len(tmp_list) < 1:
                tmp_val_index = 1
            else:
                tmp_val_index = int(val_ratio * len(tmp_list))
            tmp_test_index = int(test_ratio * len(tmp_list)) + tmp_val_index

            tmp_val_list = tmp_list[:tmp_val_index]
            tmp_test_list = tmp_list[tmp_val_index:tmp_test_index]
            tmp_train_list = tmp_list[tmp_test_index:]
            #print(len(tmp_val_list))
            #print(len(tmp_train_list))
            #print(len(tmp_test_list))
            #assert False

            val_list.append(tmp_val_list)
            train_list.append(tmp_train_list)
            test_list.append(tmp_test_list)
        

        return (train_list, val_list, test_list)




        #print(len(instance_list))
        #val_len = int(0.1 * len(instance_list))
        #text_len = val_len
        #val_list = random.sample(instance_list, 5600)
        #for tmp_dict in val_list:
        #    json.dump(eval(tmp_dict), outfile)
        #    outfile.write("\n")


if __name__ == "__main__":
    val_ratio = float(sys.argv[1])
    test_ratio = float(sys.argv[2])
    train_ratio = round(1 - val_ratio - test_ratio, 4)
    infile = sys.argv[3]
    data_name = sys.argv[4]

    modes = ["train", "val", "test"]

    splited_dict = dict()
    splited_dict["train"], splited_dict["val"], splited_dict["test"] = split(infile, val_ratio, test_ratio)

    for mode in modes:
        with open("{}_{}/{}_{}_{}.txt".format(data_name, train_ratio, data_name,train_ratio,  mode), "w") as outfile:
            _list =  splited_dict["{}".format(mode)]
            print(len(_list))
            tmp_list=list(itertools.chain.from_iterable(_list))
            random.shuffle(tmp_list)
            print(len(tmp_list))
            for tmp_dict in tmp_list:
                json.dump(tmp_dict, outfile)
                outfile.write("\n")


                






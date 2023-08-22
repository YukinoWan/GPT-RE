import json
import sys
from shared.prompt import instance


def prompt(task, x):
    if task == "semeval":
        instruction = "Given the context, please choose the most accurate relation of the subject towards the object from the following relation types: Component-Whole, Cause-Effect, Entity-Destination, Entity-Origin, Product-Producer, Member-Collection, Message-Topic, Content-Container, Instrument-Agency. If the relation does not belong to any of these above, please output Other."

    input_str = x.context + "Subject: " + x.head + ".\n" + "Object: " + x.tail +"."

    output_str = x.rel

    dict_ = {}
    dict_["instruction"] = instruction
    dict_["input"] = input_str
    dict_["output"] = output_str

    return dict_


def transfer(input_file, out_file, task):
    with open(out_file, "w") as outfile:
        file_list = []
        with open(input_file, "r") as f:
            print("OK")

            for line in f.read().splitlines():
                tmp_dict = json.loads(line)
                x = instance(tmp_dict)

                prompt_dict = prompt(task, x)
                file_list.append(prompt_dict)
        json.dump(file_list, outfile, indent=2)







if __name__ == "__main__":
    input_file = sys.argv[1]
    out_file = sys.argv[2]
    task = sys.argv[3]

    transfer(input_file, out_file, task)

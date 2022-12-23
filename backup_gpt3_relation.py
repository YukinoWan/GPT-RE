import json
import os
import pandas as pd
import argparse
import sys
import math
from gpt3_api import Demo
import random
import numpy as np
from testeval import compute_f1
from shared.const import semeval_reltoid
from shared.const import semeval_idtoprompt
from shared.const import ace05_reltoid
from shared.const import ace05_idtoprompt
from sklearn.metrics import classification_report
from knn_simcse import find_knn_example
from simcse import SimCSE

 
def generate_relation_dict_label(dataset):
    labels = []
    with open(dataset, "r") as f:
        relation_dict = {}
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "None"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return relation_dict, labels
def generate_label(dataset, relation_dict):
    labels = []
    with open(dataset, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "NONE"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return labels



def generate_query(h_type, t_type, relation_list, query_dict):
    query_list = []
    #print(query_dict)
    for rel in relation_list:
        if rel == "None":
            continue
        else:
            query = query_dict[str((h_type,rel,t_type))]
            query_list.append(query)
    return query_list



def build_query_dict(dataset):
    with open("query_templates/ace2005.json", "r") as f:
        whole_dict = json.load(f)
        query_dict = whole_dict["qa_turn2"]
        return query_dict



def generate_zero_prompt(tmp_dict, query_dict, relation_list):
    prompt_list = []
    if True:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        query_list = generate_query(entity1_type, entity2_type, relation_list, query_dict)
        for query in query_list:
            prompt = "Context: " + string + "\n" + "Please " + query.replace("XXX", entity1) + "\nEntities:"
            prompt_list.append(prompt)

        return prompt_list , entity1, entity2
def generate_select_prompt(tmp_dict, query_dict, relation_list):
    #prompt_list = []
    if True:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        prompt = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output the number 0.\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:"
        #print(prompt)
        #prompt_list.append(prompt)

        return prompt, entity1, entity2

def generate_select_auto_prompt(tmp_dict, example_prompt, relation_dict, no_na, reasoning):
    #prompt_list = []
    if True:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        #query_list = generate_query(entity1_type, entity2_type, relation_list, query_dict)
        #for query in query_list:
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output the number 0.\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\n"
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        choice_def = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
        choice_reason = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT AND PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINE\nINSTRUMENT AND AGENCY\n"

        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\n"
        task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to nine possible relations. If yes, I will output the most precise relation, otherwise I will output OTHER.\n\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\nCAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
        task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities based on the context, choosing from nine possible relations:\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all relations are not proper, I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        task_def_others = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from seven possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\nOTHERS: the relation does not belongs to the previous six choices\n"
        task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output None."
        query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is"
        #query = "Given the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be"
        #query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput:"

        if reasoning:
            choice = choice_def
        else:
            choice = choice_def
        if no_na:
            prompt = task_def_choice + choice + example_prompt + query
        else:

            prompt = task_def_choice_na + choice + example_prompt + query
        #print(prompt)
        #assert False
        #assert False
        #prompt_list.append(prompt)

        return prompt, entity1, entity2

def get_train_example(example_path, reltoid, no_na):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict
     
def get_test_example(example_path, reltoid):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict
     

def auto_generate_example(example_dict, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo):
    #ratio = 0.5
    #num_per_rel = 4
    num_example = num_per_rel * (len(example_dict.keys()) - 1) + num_na


    #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #reltoalpha = {0:"0", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    #reltoalpha = {0:"NONE", 1:"Physical", 2:"General and affiliation", 3:"Person and social", 4:"Organization and affiliation", 5:"Part and whole", 6:"Agent and artifact"}
    #reltoalpha = {0:"NONE", 1:"PHYSICAL", 2:"GENERAL AND AFFILIATION", 3:"PERSON AND SOCIAL", 4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}
           #else:
            #    if random.random() > 0.9:
            #        example_list.append(tmp_dict)
            #    else:
            #        continue
    #examples = [item for k,v in example_dict.items() for item in v]
    examples = []
    for relid in example_dict.keys():
        find_knn_example(knn_model, tmp_dict,train_dict,k, var, no_na)
    
    example_prompt = str()
    #if var and no_na and label_other:
    #    return 

    tmp_knn = []
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        if not reasoning:
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:"
            #print(prompt_query)
            #assert False

            results, probs = demo.get_multiple_sample(tmp_query)
            prompt_query = prompt_query + results[0] +"\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other



def run(reltoid, idtoprompt, store_path, args):
    demo = Demo(
            engine=args.model,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            logprobs=1,
            )
    #relation_dict = {'None': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}relation_dict = {'Others': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #reltoid = {'NONE': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #idtoprompt = {0: "NONE", 1: "PHYSICAL", 2: "GENERAL AND AFFILIATION", 3: "PERSON AND SOCIAL", 4: "ORGANIZATION AND AFFILIATION", 5: "PART AND WHOLE", 6: "AGENT AND ARTIFACT"}
    #relation_dict = {'OTHERS': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #query_dict = build_query_dict(dataset)
    #all_labels = generate_label(dataset, reltoid)

    example_dict = get_train_example(args.example_dataset, reltoid, args.no_na)
    test_dict = get_test_example(args.test_dataset, reltoid)
    flat_examples = [item for sublist in test_dict.values() for item in sublist]
    test_examples = random.sample(flat_examples, args.num_test)

    if args.use_knn:
        #train_list = test_examples
        train_list = [x for y in example_dict.values() for x in y]
        if args.no_na:
            train_list = [x for x in train_list if reltoid[x["relations"][0][0][4]] != 0]
        train_dict = {" ".join(x["sentences"][0]):x for x in train_list}
        train_sentences = [x for x in train_dict.keys()]

        #knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        knn_model.build_index(train_sentences, device="cpu")

    print(len(test_examples))

    micro_f1 = 0.0
    #example_prompt = auto_generate_example(example_dataset, relation_dict, 18, True)
    for run in range(args.num_run):
        example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo)
        print(example_prompt)
        if not args.fixed_test:
            test_examples = random.sample(flat_examples, args.num_test)
        labels = []
        preds = []
        num = 0
        whole_knn = []
        for tmp_dict in test_examples:
            tmp_knn = []
            #tmp_dict = json.loads(line)
            na_filter = random.random()
            #rel_filter = random.random()
            if tmp_dict["relations"] == [[]] and args.no_na:
                num += 1
                continue
            elif tmp_dict["relations"] == [[]] and na_filter < 0.95:
                num += 1
                continue
            if tmp_dict["relations"] != [[]] and tmp_dict["relations"][0][0][4] == "Other" and args.no_na:
                num += 1
                continue
            #if rel_filter < 0.5:
            #    lineid += 1
            #    continue

            #example_dict = get_train_example(example_dataset, reltoid)
            if not args.fixed_example and not args.use_knn:
                example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo)
            if args.use_knn:
                example_prompt, tmp_knn, label_other = generate_knn_example(knn_model, tmp_dict, train_dict, args.k, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo, args.var, args.no_na)
                whole_knn.append(tmp_knn)
            num += 1
            if tmp_dict["relations"] == [[]]:
                labels.append(0)
            else:
                labels.append(reltoid[tmp_dict["relations"][0][0][4]])
            sentence = " ".join(tmp_dict["sentences"][0])
            #prompt_list, subject, target = generate_zero_prompt(tmp_dict, query_dict, relation_dict.keys())

            if not label_other:
                prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.no_na, args.reasoning)
                pred, prob_on_rel = get_results_select(demo, prompt_list, idtoprompt)
            else:
                pred = 0
            preds.append(pred)
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            
            if preds[-1] != labels[-1]:
                with open("{}/negtive.txt".format(store_path), "a") as negf:
                
                    #negf.write(args)
                    #negf.write("\n")
                    negf.write(prompt_list + "\n")
                    
                    negf.write(str(reltoid) + "\n")
                    negf.write(str(prob_on_rel) + "\n")
                    negf.write("Prediction: " + str(preds[-1]) + "\n")
                    #negf.write(preds[num])
                    negf.write("Gold: " + str(labels[-1]) + "\n")
                    #negf.write(labels[num])
                    negf.write("\n-----------------\n")

            with open("{}/results.txt".format(store_path),"a") as negf:
                #negf.write(args)
                #negf.write("\n")
                negf.write(prompt_list + "\n")
                    
                negf.write(str(reltoid) + "\n")
                negf.write(str(prob_on_rel) + "\n")
                negf.write("Prediction: " + str(preds[-1]) + "\n")
                #negf.write(preds[num])
                negf.write("Gold: " + str(labels[-1]) + "\n")
                #negf.write(str(classification_report(labels[:num], preds, digits=4)))
                negf.write(str(f1_result))
                negf.write("\n")
                #negf.write(labels[num])
                negf.write("\n-----------------\n")
            #print(results[0])
            #print(probs[0])
            #if num > 100:
            #    assert False
            print("processing:", 100*num/len(test_examples), "%", end="\n")
            print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, digits=4,output_dict=True)
        with open("{}/labels.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(labels)]))
        with open("{}/preds.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(preds)]))
        micro_f1 += f1_result["f1"]
        with open("{}/knn.csv".format(store_path), "w") as f:
            for line in whole_knn:
                f.write('\n'.join([str(line)]))
                f.write("\n")
        df = pd.DataFrame(report).transpose()
        df.to_csv("{}/result_per_rel.csv".format(store_path))
        #assert False
    avg_f1 = micro_f1 / args.num_run
    print("AVG f1:", avg_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True, choices=["ace05","semeval"])
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--example_dataset", type=str, default=None, required=True)
    parser.add_argument("--test_dataset", type=str, default=None, required=True)
    parser.add_argument("--fixed_example", type=int, default=1)
    parser.add_argument("--fixed_test", type=int,default=1)
    parser.add_argument("--num_per_rel", type=int, default=2)
    parser.add_argument("--num_na", type=int, default=0)
    parser.add_argument("--no_na", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_label", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--use_knn", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--bert_sim", type=int, default=1)
    parser.add_argument("--var", type=int, default=0)


    args = parser.parse_args()
    if args.var:
        args.var = True
    else:
        args.var = False
    if args.fixed_example and args.use_knn:
        assert False
    if args.fixed_example == 1:
        args.fixed_example = True
    else:
        args.fixed_example = False

    if args.fixed_test == 1:
        args.fixed_test = True
    else:
        args.fixed_test = False

    if args.reasoning == 1:
        args.reasoning = True
    else:
        args.reasoning = False

    if args.no_na == 1:
        args.no_na = True
    else:
        args.no_na = False

    if args.random_label == 1:
        args.random_label = True
    else:
        args.random_label = False
    print(args)
    if args.no_na and args.num_na != 0:
        print(args.no_na)
        print(args.num_na)
        assert False
    store_path = "./knn_{}_results/subtest_1/knn={}_var={}_{}_{}_noNA={}_seed={}_{}_randomlabel={}_fixedex={}_fixedtest={}".format(args.task, args.k, args.var, args.num_per_rel,args.num_na,args.no_na, args.seed,args.model,str(args.random_label),str(args.fixed_example),str(args.fixed_test), str(args.reasoning))
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    
    #task = sys.argv[1]
    #test_num = int(sys.argv[2])
    #seed = sys.argv[3]
    random.seed(args.seed)
    if args.task == "semeval":
        #example_dataset = "./dataset/semeval_gpt/train.json"
        #dataset = "./dataset/semeval_gpt/test.json"
        run(semeval_reltoid,semeval_idtoprompt, store_path, args)
    else:
        #example_dataset = "./dataset/ace05/test.json"
        #dataset = "./dataset/ace05/ace05_0.2/ace05_0.2_test.txt"
        run(ace05_reltoid,ace05_idtoprompt, store_path, args)
    #relation_list = ["\""+x+"\"" for x in tacred_relation.keys()]
    #relation_set = ",".join(relation_list)

    #nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

                
               
    #results = [x.strip().strip("\"") for x in results[0]]
    #print(results[0])
    #print(rel)
    #if False:
    #    if results[0].strip() in tacred_relation.keys():
    #    preds.append(tacred_relation[results[0].strip()])
    #        print(rel)
    #                if rel == results[0].strip():
    #                    print("OK!")
    #                else:
    #                    with open("./negative_results.txt","a") as negf:
    #                        negf.write(neg_prompt)
    #                        negf.write("\nPrediction:")
    #                        negf.write(results[0])
    #                        negf.write("\nGold:")
    #                        negf.write(rel)
    #                        negf.write("\n-----------------\n")
    #            else:
    #                print("None")
    #                preds.append(0)
                #print(preds)
                #print(labels)
                #assert False
    #            result = compute_f1(preds, labels)
    #            print(result, end="\n")
                #assert False
    
                #dependency(nlp, string)


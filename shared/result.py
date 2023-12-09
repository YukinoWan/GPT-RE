import json
import argparse
import sys
import math
from gpt3_api import Demo
import random
import numpy as np
from testeval import compute_f1



def find_prob(target, result, probs):
    if False:
        print(result)
        print("targettarget\n")
        print(target)
        print("tokentoken\n")
        print(probs["tokens"])
        print("===============\n")
    try:
        index = [x.strip() for x in probs["tokens"]].index(str(target))
        #print(probs["token_logprobs"][index])
        return math.exp(probs["token_logprobs"][index])
    except:
        len_target = len(target)
        for i in range(2, len_target+1):
            for j in range(len(probs["tokens"])):
                if i + j > len(probs["tokens"]):
                    continue
                #print(j+i)
                #print(len(probs["tokens"]))
                tmp_word = "".join([probs["tokens"][x] for x in range(j, j+i)])
                if tmp_word.strip() != target:
                    #print(tmp_word.strip())
                    continue
                else:
                    #print(tmp_word.strip())

                    start = j
                    end = j + i
                    sum_prob = 0
                    for k in range(start, end):
                        sum_prob += math.exp(probs["token_logprobs"][k])
                        #print(sum_prob)
                    #if sum_prob == None:
                        #print(target)
                        #print(result)
                    return sum_prob / i
        return 0.0
def smooth(x):
    if True:
        return np.exp(x)/sum(np.exp(x)) 
    else:
        return x


def get_results_onebyone(demo, prompt_list, target):
    threshold = 0.2
    prob_on_rel = []
    for prompt in prompt_list:
        results, probs = demo.get_multiple_sample(prompt)
        #print("----------------\n")
        #print(prompt)
        #print(results[0])
        #print(probs[0])
        
        if target in results[0]:
            #probs_list = {k.strip():data[k] for data in probs[0]["top_logprobs"] for k in data.keys()}
            #print(probs_list)
            #print("promp")
            #print(prompt)
            prob = find_prob(target, results[0], probs[0])
            prob_on_rel.append(prob)
            #print(prob_on_rel)
        else:
            prob_on_rel.append(0)
            #print(prob_on_rel)
    prob_on_rel = np.insert(smooth(np.array(prob_on_rel)), 0, threshold)
    pred = np.argmax(prob_on_rel)
    return pred, prob_on_rel



def get_results_select(demo, prompt, reltoid, idtoprompt_ori, verbalize, args):
    #idtoprompt = {k:v for k,v in idtoprompt_ori.items()}
    #print(prompt)
    #assert False
    while True:
        try:
            results, probs = demo.get_multiple_sample(prompt)
            break
        except:
            return 0,0,0, True

    if args.task == "semeval":
        idtoprompt = {"none":0, "cause and effect: an event or object yields an effect":1,"component and whole: an object is a component of a larger whole":2,"entity and destination: an entity is moving towards a destination":3,"entity and origin: an entity is coming or is derived from an origin":4,"product and producer: a producer causes a product to exist":5,"member and collection: a member forms a nonfunctional part of a collection":6, "message and topic: an act of communication, writter or spoken, is about a topic":7,"content and container: an object is physically stored in a delineated area of space":8, "instrument and agency: an agent uses an instrument":9}
    else:
        idtoprompt = reltoid
    if verbalize:
        idtoprompt[4] += "/PER:NATIONALITY/PER:ETHNICITY"
        idtoprompt[3] += "/ORG:MERGERS"
        idtoprompt[30] += "/PER:CITY_OF_RESIDENCE"
        idtoprompt[1] += "/PER:OCCUPATION"
        idtoprompt[19] += "/ORG:ALTERNATE_NAME"
        idtoprompt[32] += "/PER:CRIMINAL_CHARGE"
        idtoprompt[6] += "/ORG:LOCATION_OF_HEADQUARTERS"
        idtoprompt[9] += "/PER:EMPLOYER"
        idtoprompt[5] += "/ORG:EMPLOYEES/ORG:EMPLOYERS/ORG:EMPLOYER/ORG:EMPLOYEE"
    #select_dict = {"none":0, "physical: located, near":1,"general and affiliation: citizen, resident, religion, ethnicity, organization location":2,"person and social: business,family,lasting personal":3,"organization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership":4,"part and whole: artifact,geographical,subsidiary":5,"agent and artifact: user, owner, inventor, manufacturer":6}
 
    if True:
        #return int(select_dict[results[0].strip()]), math.exp(probs[0]["token_logprobs"][0])
        #choice = [select_dict[i] for i in select_dict.keys() if results[0].strip() in select_dict[i]]]
        choice = 0
        for key, value in idtoprompt.items():
            if results[0].strip().strip(".").lower() in key.lower():
                choice = value
        if choice == 0:
            for key in idtoprompt_ori.keys():
                if idtoprompt_ori[key].lower() in results[0].lower():
                    choice = key
        #assert False
        print(results)
        #print(probs)
        print("the choice is ",choice)
        #if int(choice) == 7:
        #    print(results)
        #    assert False
        #print(choice)
        return int(choice), math.exp(probs[0]["token_logprobs"][0]), probs[0], False
    else:
        print(prompt)
        print(results[0].strip())
        print(probs[0]["token_logprobs"][0])
        assert False

    

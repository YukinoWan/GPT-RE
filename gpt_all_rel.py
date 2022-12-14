import json
import math
from gpt3_api import Demo
import random
import numpy as np
from testeval import compute_f1
from tacred_rel2id import tacred_relation
from sklearn.metrics import classification_report

#def dependency(nlp, string):
   #doc = nlp(string)
   #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')o
def generate_sample_prompt(tmp_dict, mode):
    prompt1 = "Tom Thabane resigned in October last year to form the All Basotho Convention -LRB- ABC -RRB- , crossing the floor with 17 members of parliament , causing constitutional monarch King Letsie III to dissolve parliament and call the snap election . \nIn the above sentence, \nQ:\"All Basotho\" is founded by \"Tom Thabane\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is founded by \"John D.\"?\nA:false\n"
    prompt2 = "Argentina on Thursday announced it had regained control of the national flagship carrier Aerolineas Argentinas after signing a deal with the principal shareholder , the Spanish group Marsans .\nIn the above sentence,\nQ:\"Aerolineas Argentinas\" is the shareholder with \"Marsans\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is the shareholder with \"John D.\"?\nA:false\n"
    prompt3 = "Sadia said it fired chief financial officer Adriano Ferreira , and Aracruz CFO Isac Zagury also offered his resignation -- although the company would not say if it had accepted it .nIn the above sentence,\nQ:\"Sadia\" is a member of \"Aracruz\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is a member of \"John D.\"?\nA:false\n"
    prompt4 = "Please select one relation from the set {" + relation_set + "} that best describes the relationship between two entities, if no specific relation is proper, please select \"None\".\n"


                #tmp_prompt = prompt3 + string + "\n" + "In the above sentence,\n" + "Q:\"" + entity1 +"\" is a member of\"" + entity2 + "\"?\nA:"
    #tmp_prompt = prompt4 + string + "\n" + "In the above sentence,\n" + "The relationship between \"" + entity1 +"\" and \"" + entity2 + "\" is:"
    #neg_prompt = string + "\n" + "In the above sentence,\n" + "The relationship between \"" + entity1 +"\" and \"" + entity2 + "\" is:"
 
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


def generate_few_prompt(tmp_dict, query_dict, relation_list):
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
            prompt = "Context: the lady at the pentagon saying she was leaving , saying it had been an honor serving her post but she 's leaving for personal reasons .\nPlease find all organization entities in the context that have an organization affiliation relationship with person entity lady.\nEntities: pentagon\n"
            prompt = "Context: " + string + "\n" + "Please " + query.replace("XXX", entity1) + "\nEntities:"
            prompt_list.append(prompt)

        return prompt_list , entity1, entity2


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

        #query_list = generate_query(entity1_type, entity2_type, relation_list, query_dict)
        #for query in query_list:
        #prompt = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context and two entities, I'll output the most precise relation of the two entities in the context, choosing from three possible relations. I will output one number of the most precise choice. If all choices are not proper, I will output number 0.\nContext: " + string + "\n" + "Entity1: " + entity1 + "\nEntity2: " + entity2 + "\nChoice 1: physical relation\nChoice 2: gen and affiliation relation\nChoice 3: person and social relation\nChoice 4: organization and affiliation relation\nChoice 5: part and whole relation\nChoice 6: artifact relation\nOutput:"
        prompt = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output the number 0.\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:"
        #print(prompt)
        #prompt_list.append(prompt)

        return prompt, entity1, entity2
def generate_select_few_prompt(tmp_dict, query_dict, relation_list):
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
        task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output the number 0.\n"
        prompt1 = "\nContext: And when you arrive there , you are greeted in a sense by Saddam Hussein .\nSubject: you\nObject:there\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:A\n"
        prompt2 = "\nContext: it hurts to think of how many wonderful people in iraq at the academic community , lawyers , human rights activists , civilians , intellectuals , people -- would now be alive .\nSubject: people\nObject:iraq\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:B\n"
        prompt3 = "\nContext: Holding a framed picture of her son , serving with the Army 's 3rd Infantry Division in Iraq , she said she did n't know whether he was dead or alive . \nSubject: her\nObject:son\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:C\n" 
        prompt4 = "\nContext: reporter : the findings are heightening concerns among coalition forces approaching baghdad since , as the iraqis should know , u.s . and british forces do not use chemical weapons .\nSubject: forces\nObject:coalition\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:D\n" 
        prompt5 = "\nContext: life on the homefront seems the same at farmer 's market in los angeles , but it 's more angst ridden .\nSubject: market\nObject:los angeles\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:E\n" 
        prompt6 = "\nContext: reporter : and she reveals success in a 1989 test , using aerial bombs to disperse biological agents .\nSubject: she\nObject:agents\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:F\n" 
        prompt0 = "\nContext: I should disclose that I spent a few years living in Seattle and I have a soft spot in my heart for all things Seattle .\nSubject: I\nObject:my\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:0\n" 
        query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\nOutput:"

        prompt = task_def + prompt1 + prompt2 +prompt3+prompt4+prompt5+prompt6 + query
        print(prompt3)
        assert False
        #assert False
        #prompt_list.append(prompt)

        return prompt, entity1, entity2


def generate_select_auto_prompt(tmp_dict, example_prompt, relation_dict):
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
        task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        task_def_others = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from seven possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\nOTHERS: the relation does not belongs to the previous six choices\n"
        task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output None."
        query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput:"

        prompt = task_def_choice + example_prompt + query
        #print(prompt)
        #assert False
        #assert False
        #prompt_list.append(prompt)

        return prompt, entity1, entity2
def auto_generate_example(example_dataset, relation_dict, num_example, No_None):
    example_dict = {k:list() for k in relation_dict.keys()}
    #ratio = 0.5
    num_per_rel = 4
    num_ = 0


    #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #reltoalpha = {0:"0", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    #reltoalpha = {0:"NONE", 1:"Physical", 2:"General and affiliation", 3:"Person and social", 4:"Organization and affiliation", 5:"Part and whole", 6:"Agent and artifact"}
    reltoalpha = {0:"NONE", 1:"PHYSICAL", 2:"GENERAL AND AFFILIATION", 3:"PERSON AND SOCIAL", 4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}
    with open(example_dataset, "r") as f:
        for line in f.read().splitlines():
            if num_ == num_example:
                break
            tmp_dict = json.loads(line)
            if No_None and tmp_dict["relations"] == [[]]:
                continue
            elif tmp_dict["relations"] == [[]]:
                rel = "NONE"
                if len(example_dict[rel]) < num_example - num_per_rel * 6:
                    example_dict[rel].append(tmp_dict)
                    num_ += 1
                else:
                    continue
            else:
                rel = tmp_dict["relations"][0][0][4]

                if len(example_dict[rel]) < num_per_rel:
                    example_dict[rel].append(tmp_dict)
                    num_ += 1
                else:
                    continue
            #else:
            #    if random.random() > 0.9:
            #        example_list.append(tmp_dict)
            #    else:
            #        continue
    examples = [item for k,v in example_dict.items() for item in v]
    #print(len(examples))
    example_list = random.sample(examples, len(examples))
    #assert False
    
    example_prompt = str()
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

        if tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]


        #rel = random.choice(["None", "PHYS","GEN-AFF", "PER-SOC","ORG-AFF","PART-WHOLE", "ART"])

        #query = "\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship"
        query = "\nPhysical: located, near\nGeneral and affiliation: citizen, resident, religion, ethnicity, organization location\nPerson and social: business,family,lasting personal\nOrganization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPart and whole: artifact,geographical,subsidiary\nAgent and artifact: user, owner, inventor, manufacturer"
        #query = "\nPhysical\nGeneral and affiliation\nPerson and social\nOrganization and affiliation\nPart and whole\nAgent and artifact"

        #prompt_query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + query + "\nOutput: " + reltoalpha[relation_dict[rel]]
        prompt_query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput: " + reltoalpha[relation_dict[rel]] + "\n"
        example_prompt += prompt_query
    return example_prompt

def auto_other_generate_example(example_dataset, relation_dict, num_example, No_None):
    example_dict = {k:list() for k in relation_dict.keys()}
    #ratio = 0.5
    num_per_rel = 2
    num_ = 0


    #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #reltoalpha = {0:"0", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    #reltoalpha = {0:"None", 1:"Physical", 2:"General and affiliation", 3:"Person and social", 4:"Organization and affiliation", 5:"Part and whole", 6:"Agent and artifact"}
    reltoalpha = {0:"OTHERS", 1:"PHYSICAL", 2:"GENERAL AND AFFILIATION", 3:"PERSON AND SOCIAL", 4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}
    with open(example_dataset, "r") as f:
        for line in f.read().splitlines():
            if num_ == num_example:
                break
            tmp_dict = json.loads(line)
            if No_None and tmp_dict["relations"] == [[]]:
                continue
            elif tmp_dict["relations"] == [[]]:
                rel = "OTHERS"
                #rel = "None"
                if len(example_dict[rel]) < num_example - num_per_rel * 6:
                    example_dict[rel].append(tmp_dict)
                    num_ += 1
                else:
                    continue
            else:
                rel = tmp_dict["relations"][0][0][4]

                if len(example_dict[rel]) < num_per_rel:
                    example_dict[rel].append(tmp_dict)
                    num_ += 1
                else:
                    continue
            #else:
            #    if random.random() > 0.9:
            #        example_list.append(tmp_dict)
            #    else:
            #        continue
    print(len(examples))
    assert False
    examples = [item for k,v in example_dict.items() for item in v]
    example_list = random.sample(examples, len(examples))
    
    example_prompt = str()
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

        if tmp_dict["relations"] == [[]]:
            rel = 'OTHERS'
        else:
            rel = tmp_dict["relations"][0][0][4]


        #rel = random.choice(["None", "PHYS","GEN-AFF", "PER-SOC","ORG-AFF","PART-WHOLE", "ART"])

        #query = "\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship"
        #query = "\nPhysical: located, near\nGeneral and affiliation: citizen, resident, religion, ethnicity, organization location\nPerson and social: business,family,lasting personal\nOrganization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPart and whole: artifact,geographical,subsidiary\nAgent and artifact: user, owner, inventor, manufacturer\nOthers: the relation does not belongs to the previous types"
        #query = "\nPhysical\nGeneral and affiliation\nPerson and social\nOrganization and affiliation\nPart and whole\nAgent and artifact"

        #prompt_query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + query + "\nOutput: " + reltoalpha[relation_dict[rel]]
        prompt_query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput: " + reltoalpha[relation_dict[rel]] + "\n"
        example_prompt += prompt_query
    return example_prompt



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



def get_results_select(demo, prompt):
    #print(prompt)
    #assert False
    results, probs = demo.get_multiple_sample(prompt)
    #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #select_dict = {"others":0, "physical: located, near":1,"general and affiliation: citizen, resident, religion, ethnicity, organization location":2,"person and social: business,family,lasting personal":3,"organization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership":4,"part and whole: artifact,geographical,subsidiary":5,"agent and artifact: user, owner, inventor, manufacturer":6}
    select_dict = {"none":0, "physical: located, near":1,"general and affiliation: citizen, resident, religion, ethnicity, organization location":2,"person and social: business,family,lasting personal":3,"organization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership":4,"part and whole: artifact,geographical,subsidiary":5,"agent and artifact: user, owner, inventor, manufacturer":6}
    #select_dict = {"none":0, "physical: located, near":1,"general and affiliation: citizen, resident, religion, ethnicity, organization location":2,"person and social: business,family,lasting personal":3,"organization and affiliation: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership":4,"part and whole: artifact,geographical,subsidiary":5,"agent and artifact: user, owner, inventor, manufacturer":6}
    #print(results[0].strip())
    #print(probs[0]["token_logprobs"][0])
    #assert False
    try:
        #return int(select_dict[results[0].strip()]), math.exp(probs[0]["token_logprobs"][0])
        #choice = [select_dict[i] for i in select_dict.keys() if results[0].strip() in select_dict[i]]]
        choice = 0
        for key in select_dict.keys():
            if results[0].strip().lower() in key:
                choice = select_dict[key]
        
        if int(choice) == 7:
            print(results)
            assert False
        #print(choice)
        return int(choice), math.exp(probs[0]["token_logprobs"][0])
    except:
        print(prompt)
        print(results[0].strip())
        print(probs[0]["token_logprobs"][0])
        assert False

    

def run(example_dataset, dataset):
    #relation_dict = {'None': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}relation_dict = {'Others': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    relation_dict = {'NONE': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #relation_dict = {'OTHERS': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    query_dict = build_query_dict(dataset)
    all_labels = generate_label(dataset, relation_dict)
    labels = []
    preds = []
    num = 0
    lineid = 0
    #example_prompt = auto_generate_example(example_dataset, relation_dict, 18, True)
    with open(dataset, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            na_filter = random.random()
            rel_filter = random.random()
            if tmp_dict["relations"] == [[]]:
                lineid += 1
                continue
            #if rel_filter < 0.5:
            #    lineid += 1
            #    continue

            example_prompt = auto_generate_example(example_dataset, relation_dict, 24, True)
            labels.append(all_labels[lineid])
            lineid += 1
            sentence = " ".join(tmp_dict["sentences"][0])
            #prompt_list, subject, target = generate_zero_prompt(tmp_dict, query_dict, relation_dict.keys())
            prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, relation_dict)
            demo = Demo(
                    engine="text-davinci-003",
                    temperature=0,
                    max_tokens=20,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    best_of=1,
                    logprobs=1,
                    )
            #results, probs = demo.get_multiple_sample(prompt_list)
            #pred, prob_on_rel = get_results_onebyone(demo, prompt_list, target)
            print(prompt_list)
            assert False
            pred, prob_on_rel = get_results_select(demo, prompt_list)
            preds.append(pred)
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            
            if preds[num] != labels[num]:
                with open("./log_file_few_reltype/12nona_shot_selectpredefined_ace05_davinci_negtive.txt", "a") as negf:
                
                    negf.write(sentence + "\n")
                    negf.write("subject:" + subject + "\n")
                    negf.write("object:" + target + "\n")
                    negf.write(str(prompt_list) + "\n")
                    negf.write(str(relation_dict) + "\n")
                    negf.write(str(prob_on_rel) + "\n")
                    negf.write("Prediction: " + str(preds[num]) + "\n")
                    #negf.write(preds[num])
                    negf.write("Gold: " + str(labels[num]) + "\n")
                    #negf.write(labels[num])
                    negf.write("\n-----------------\n")

            with open("./log_file_few_reltype/12nona_shot_selectpredefined_ace05_davinci_results.txt","a") as negf:
                negf.write(sentence + "\n")
                negf.write("subject:" + subject + "\n")
                negf.write("object:" + target + "\n")
                negf.write(str(prompt_list) + "\n")
                negf.write(str(relation_dict) + "\n")
                negf.write(str(prob_on_rel) + "\n")
                negf.write("Prediction: " + str(preds[num]) + "\n")
                #negf.write(preds[num])
                negf.write("Gold: " + str(labels[num]) + "\n")
                #negf.write(str(classification_report(labels[:num], preds, digits=4)))
                negf.write(str(f1_result))
                negf.write("\n")
                #negf.write(labels[num])
                negf.write("\n-----------------\n")
            #print(results[0])
            #print(probs[0])
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            num += 1
            print(classification_report(labels[:num], preds, digits=4))
            if num > 100:
                assert False
            print("processing:", 100*num/2548, "%", end="\n")


if __name__ == "__main__":
    random.seed(2)
    example_dataset = "./dataset/ace05/test.json"
    dataset = "./dataset/ace05/ace05_0.2/ace05_0.2_test.txt"
    run(example_dataset, dataset)
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


import json
from shared.result import get_results_select


class instance:
    def __init__(self, tmp_dict):
        self.sentence = " ".join(tmp_dict["sentences"][0])
        self.id = tmp_dict["doc_key"]
        self.rel = self.get_relation(tmp_dict)
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        self.head = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        self.head_type = tmp_dict["ner"][0][0][2]
        self.tail = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        self.tail_type = tmp_dict["ner"][0][1][2]

        self.reference = ("The relation between \"" + self.head + "\" and \""
                          + self.tail + "\" in the sentence \"" + self.sentence + "\"")
        self.lm_mask = ("The relation between \"" + self.head + "\" and \""
                          + self.tail + "\" in the sentence \"" + self.sentence + "\" is <mask>.")
        self.context = "\nContext: " + self.sentence + "\n"
        self.discriminator = "\nContext: " + self.sentence + "\n" + "Question: given the context, whether is the relation between " + self.head + " and " + self.tail + " being "
        self.answer = "\nAnswer:"
    def get_relation(self, tmp_dict):
        if tmp_dict["relations"] == [[]]:
            return "NONE"
        else:
            return tmp_dict["relations"][0][0][4]

    def get_reason(self, idtoprompt, reltoid):
        reason = ("What are the clues that lead the relation between \""
                  + self.head + "\" and \"" + self.tail + "\" to be "
                  + idtoprompt[reltoid[self.rel]] + " in the sentence \""
                  + self.sentence + "\"?")
        return reason

    def get_self_error(self, tmp_dict, demo, reltoid, idtoprompt, args):
        example_prompt = ""
        prompt_list, subject, target = generate_select_auto_prompt(
                tmp_dict, example_prompt, reltoid, 
                args.no_na, args.reasoning, args)

        pred, prob_on_rel, prob = get_results_select(
                demo, prompt_list, reltoid, idtoprompt, 
                args.verbalize, args)

        if reltoid[self.rel] == pred:
            while(True):
                try:
                    results, probs = demo.get_multiple_sample(
                            self.get_reason(idtoprompt, reltoid))
                    break
                except:
                    continue

            if args.structure:
                prompt_query = (self.prompt + self.clue
                                + results[0] + self.pred + 
                                idtoprompt[reltoid[self.rel]] + "\n")
            else:
                prompt_query = (self.context
                                + "Given the context, the relation between " 
                                + self.head + " and " + self.tail 
                                + " is " + idtoprompt[reltoid[self.rel]] 
                                + ". It is because: \n" + results[0] + "\n")
        else:
            prompt_list = generate_self_error_prompt(
                    tmp_dict, example_prompt, reltoid, 
                    args.no_na, args.reasoning, args)
            task_prompt = prompt_list[0]
            query = (task_prompt + self.context + "Given the context, "
                     +"what are the clues that lead to the relation between "
                     +self.head + " and " + self.tail + " being " + idtoprompt[reltoid[self.rel]]
                     + ", but not " + idtoprompt[pred] + " ?")
            
            
            while(True):
                try:
                    results, probs = demo.get_multiple_sample(query)
                    break
                except:
                    continue

            #print("OK")
            #print(query)
            #print(results[0])
            if args.structure:
                prompt_query = (self.prompt + self.clue
                                + results[0] + self.pred + 
                                idtoprompt[reltoid[self.rel]] + "\n")
            else:
                prompt_query = (self.context
                                + "Given the context, the relation between " 
                                + self.head + " and " + self.tail 
                                + " is " + idtoprompt[reltoid[self.rel]] 
                                + ". It is because: \n" + results[0] + "\n")
        #print(prompt_query)
        return prompt_query

    def get_correct_reason(self, demo, idtoprompt, reltoid, args):
        if True:
            while(True):
                try:
                    results, probs = demo.get_multiple_sample(
                            self.get_reason(idtoprompt, reltoid))
                    break
                except:
                    continue

            if args.structure:
                prompt_query = (self.prompt + self.clue
                                + results[0] + self.pred + 
                                idtoprompt[reltoid[self.rel]] + "\n")
            else:
                prompt_query = (self.context
                                + "Given the context, the relation between " 
                                + self.head + " and " + self.tail 
                                + " is " + idtoprompt[reltoid[self.rel]] 
                                + ". It is because: \n" + results[0] + "\n")
            return prompt_query
            

    def get_error_reason(self, pred, tmp_dict, example_prompt, demo, idtoprompt, reltoid, args):
            prompt_list = generate_self_error_prompt(
                    tmp_dict, example_prompt, reltoid, 
                    args.no_na, args.reasoning, args)
            task_prompt = prompt_list[0]
            query = (task_prompt + self.context + "Given the context, "
                     + "what are the clues that lead to the relation between "
                     + self.head + " and " + self.tail + " being " 
                     + idtoprompt[reltoid[self.rel]]
                     + ", but not " + idtoprompt[pred] + " ?")
            
            
            while(True):
                try:
                    results, probs = demo.get_multiple_sample(query)
                    break
                except:
                    continue

            #print("OK")
            #print(query)
            #print(results[0])
            if args.structure:
                prompt_query = (self.prompt + self.clue
                                + results[0] + self.pred + 
                                idtoprompt[reltoid[self.rel]] + "\n")
            else:
                prompt_query = (self.context
                                + "Given the context, the relation between " 
                                + self.head + " and " + self.tail 
                                + " is " + idtoprompt[reltoid[self.rel]] 
                                + ". It is because: \n" + results[0] + "\n")
             #print(prompt_query)
            return prompt_query

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

def generate_select_auto_prompt(tmp_dict, example_prompt, relation_dict, no_na, reasoning, args):
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
        ace_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities choosing from the following six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        ace_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to the following six possible relations. If yes, I will output the most precise relation, otherwise I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        scierc_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to the following seven possible relations. If yes, I will output the most precise relation, otherwise I will output NONE.\n\nPART-OF: a part of\nUSED-FOR: based on, models, trained on, used for\nFEATURE-OF: belong to, a feature of\nCONJUNCTION: similar role or incorporate with\nEVALUATE-FOR: evaluate for\nHYPONYM-OF: a hyponym of, a type of\nCOMPARE: comapre with others\n"
        scierc_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities choosing from the following seven possible relations.\n\nPART-OF: a part of\nUSED-FOR: based on, models, trained on, used for\nFEATURE-OF: belong to, a feature of\nCONJUNCTION: similar role or incorporate with\nEVALUATE-FOR: evaluate for\nHYPONYM-OF: a hyponym of, a type of\nCOMPARE: comapre with others\n"

        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        #choice_def = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
        choice_def = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT AND PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINER\nINSTRUMENT AND AGENCY\n"
        #choice_def = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\n"
        choice_def_na = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\nOTHER: other possible relation types excluding these nine relations"
        choice_reason = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT AND PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINE\nINSTRUMENT AND AGENCY\n"

        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\n"
        tacred_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities. If there is no relation between them, I will output NONE\n\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to nine possible relations. If yes, I will output the most precise relation, otherwise I will output NONE.\n\n"
        task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to following nine possible relations. If yes, I will output the most precise relation, otherwise I will output NONE.\n\n"
        
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities. If yes, I will output the most precise relation, otherwise I will output OTHER.\n\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\nCAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
        task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities based on the context, choosing from nine possible relations:\n"
        tacred_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities based on the context\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all relations are not proper, I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        task_def_others = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from seven possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\nOTHERS: the relation does not belongs to the previous six choices\n"
        task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output None."
        tmp_example = instance(tmp_dict)
        if args.structure:
            if reasoning:
                query = tmp_example.prompt
            else:
                query = tmp_example.prompt + tmp_example.pred
        else:
            query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is"
        #query = instance(tmp_dict).reference + " is"
        #query = "Given the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be"
        #query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput:"

        if args.task == "tacred":
            if no_na:
                prompt = tacred_def_choice + example_prompt + query
            else:
                prompt = tacred_def_choice_na + example_prompt + query
            return prompt, entity1, entity2

        elif args.task == "ace05":
            if args.no_na:
                prompt = ace_def_choice + example_prompt + query
            else:
                prompt = ace_def_choice_na + example_prompt + query
            return prompt, entity1, entity2

        elif args.task == "scierc":
            if args.no_na:
                prompt = scierc_def_choice + example_prompt + query
            else:
                prompt = scierc_def_choice_na + example_prompt + query
            return prompt, entity1, entity2
        
        if no_na:
            prompt = task_def_choice + choice +example_prompt + query
            #prompt = task_def_choice + choice + example_prompt + instance(tmp_dict).prompt
        else:

            prompt = task_def_choice_na + choice_def + example_prompt + query
            #prompt = task_def_choice + choice + exampe_prompt + instance(tmp_dict).prompt
        #print(prompt)
        #assert False
        #assert False
        #prompt_list.append(prompt)

        return prompt, entity1, entity2

def generate_self_error_prompt(tmp_dict, example_prompt, relation_dict, no_na, reasoning, args):
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
        choice_def_na = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\nOTHER: other possible relation types excluding these nine relations"
        choice_reason = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT AND PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINE\nINSTRUMENT AND AGENCY\n"

        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\n"
        tacred_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities. If there is no relation between them, I will output NONE\n\n"
        task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to nine possible relations. If yes, I will output the most precise relation, otherwise I will output OTHER.\n\n"
        
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities. If yes, I will output the most precise relation, otherwise I will output OTHER.\n\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\nCAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
        task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities based on the context, choosing from nine possible relations:\n"
        tacred_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities based on the context\n"
        #task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all relations are not proper, I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        #task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
        task_def_others = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from seven possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\nOTHERS: the relation does not belongs to the previous six choices\n"
        task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output None."
        tmp_example = instance(tmp_dict)
        #if args.structure:
        #    if reasoning:
        #        query = tmp_example.prompt
        #    else:
        #        query = tmp_example.prompt + tmp_example.pred
        #else:
        #    query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is"
        #query = instance(tmp_dict).reference + " is"
        #query = "Given the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be"
        #query = "\nContext: " + string + "\n" + "Subject: " + entity1 + "\nObject: " + entity2 + "\nOutput:"

        if args.task == "tacred":
            if no_na:
                prompt = tacred_def_choice + example_prompt
            else:
                prompt = tacred_def_choice_na + example_prompt
            return prompt, entity1, entity2
        elif args.task == "ace05":
            if args.no_na:
                prompt = ace_def_choice + example_prompt
            else:
                prompt = ace_def_choice_na + example_prompt
            return prompt, entity1, entity2
        elif args.task == "scierc":
            if args.no_na:
                prompt = scierc_def_choice + example_prompt
            else:
                prompt = scierc_def_choice_na + example_prompt
            return prompt, entity1, entity2
 
        if reasoning:
            choice = choice_def
        else:
            choice = choice_def
        if no_na:
            prompt = task_def_choice + choice +example_prompt
            #prompt = task_def_choice + choice + example_prompt + instance(tmp_dict).prompt
        else:

            prompt = task_def_choice_na + choice + example_prompt
            #prompt = task_def_choice + choice + exampe_prompt + instance(tmp_dict).prompt
        #print(prompt)
        #assert False
        #assert False
        #prompt_list.append(prompt)

        return prompt


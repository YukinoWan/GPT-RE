import json
import gpt3_re
from testeval import compute_f1
from tacred_rel2id import tacred_relation

#def dependency(nlp, string):
   #doc = nlp(string)
   #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


if __name__ == "__main__":
    relation_list = ["\""+x+"\"" for x in tacred_relation.keys()]
    relation_set = ",".join(relation_list)

    #nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    with open("./tacred/tacred_0.1/tacred_0.1_test.txt", "r") as f:
        i = 0
        labels = []
        preds = []
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                continue
                #labels.append(0)
            elif tmp_dict["relations"][0][0][4] == "org:member_of":
                labels.append(1)
            else:
                continue
                #labels.append(0)
            if True:
                i += 1
                #if i > 5:
                #    break
                string = " ".join(tmp_dict["sentences"][0])
                sub_head = tmp_dict["ner"][0][0][0]
                sub_tail = tmp_dict["ner"][0][0][1] + 1

                
                obj_head = tmp_dict["ner"][0][1][0]
                obj_tail = tmp_dict["ner"][0][1][1] + 1

                entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
                entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
                #with open("./tacred/testeval.json", "r") as rfile:


                prompt1 = "Tom Thabane resigned in October last year to form the All Basotho Convention -LRB- ABC -RRB- , crossing the floor with 17 members of parliament , causing constitutional monarch King Letsie III to dissolve parliament and call the snap election . \nIn the above sentence, \nQ:\"All Basotho\" is founded by \"Tom Thabane\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is founded by \"John D.\"?\nA:false\n"
                prompt2 = "Argentina on Thursday announced it had regained control of the national flagship carrier Aerolineas Argentinas after signing a deal with the principal shareholder , the Spanish group Marsans .\nIn the above sentence,\nQ:\"Aerolineas Argentinas\" is the shareholder with \"Marsans\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is the shareholder with \"John D.\"?\nA:false\n"
                prompt3 = "Sadia said it fired chief financial officer Adriano Ferreira , and Aracruz CFO Isac Zagury also offered his resignation -- although the company would not say if it had accepted it .nIn the above sentence,\nQ:\"Sadia\" is a member of \"Aracruz\"?\nA:true\nIn 1983 , a year after the rally , Forsberg received the so-called \" genius award \" from the John D. and Catherine T. MacArthur Foundation .\nQ:\"Forsberg\" is a member of \"John D.\"?\nA:false\n"
                prompt4 = "Please select one relation from the set {" + relation_set + "} that best describes the relationship between two entities, if no one is proper, please select \"None\".\n"


                #tmp_prompt = prompt3 + string + "\n" + "In the above sentence,\n" + "Q:\"" + entity1 +"\" is a member of\"" + entity2 + "\"?\nA:"
                tmp_prompt = prompt4 + string + "\n" + "In the above sentence,\n" + "The relationship between \"" + entity1 +"\" and \"" + entity2 + "\" is:"
                print(tmp_prompt)
                #results,probs = run(tmp_prompt)
                demo = gpt3_re.Demo(
                    engine="text-davinci-002",  
                    temperature=0.0,  
                    max_tokens=10,  
                    top_p=1,  
                    frequency_penalty=0,  
                    presence_penalty=0,  
                    best_of=1,
                    logprobs=1
                )
                results = demo.get_multiple_sample(tmp_prompt)
                print(results)
                results = [x.strip().strip("\"") for x in results[0]]
                if results[0].strip() == "true":
                    preds.append(1)
                else:
                    print(results)
                    preds.append(0)
                #print(preds)
                #print(labels)
                #assert False
                result = compute_f1(preds, labels)
                print(result, end="\n")
                assert False
    
                #dependency(nlp, string)


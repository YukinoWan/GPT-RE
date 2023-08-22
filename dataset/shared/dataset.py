
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
        self.context = "\nContext: " + self.sentence
        self.query = ("\nQuestion: What is the relation between " + self.head
                      + " and " + self.tail + "?")
        self.clue = "\nClues: "
        self.pred = "\nAnswer: "
        self.prompt = self.context + self.query

    def get_relation(self, tmp_dict):
        if tmp_dict["relations"] == [[]]:
            return "NONE"
        else:
            return tmp_dict["relations"][0][0][4]
    
    def get_reason(self, idtoprompt, reltoid):
        reason = ("What are the clues that lead the relation between \""
                  + self.head + "\" and \"" + self.tail + "\" to be "
                  + idtoprompt[reltoid[self.rel]] + " in the sentence \"" + string + "\"?")
        return reason




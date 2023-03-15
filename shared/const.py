from shared.wiki20m_relation import wiki20m_rel

wiki_reltoid = {wiki20m_rel[i]:int(i+1) for i in range(len(wiki20m_rel))}
tacred_reltoid = {"NONE": 0, "per:title": 1, "per:city_of_death": 2, "org:shareholders": 3, "per:origin": 4, "org:top_members/employees": 5, "org:city_of_headquarters": 6, "per:religion": 7, "per:city_of_birth": 8, "per:employee_of": 9, "per:date_of_death": 10, "per:other_family": 11, "org:website": 12, "per:cause_of_death": 13, "org:subsidiaries": 14, "org:stateorprovince_of_headquarters": 15, "per:countries_of_residence": 16, "per:siblings": 17, "per:stateorprovinces_of_residence": 18, "org:alternate_names": 19, "per:spouse": 20, "per:parents": 21, "org:country_of_headquarters": 22, "per:age": 23, "per:date_of_birth": 24, "per:country_of_death": 25, "per:schools_attended": 26, "org:member_of": 27, "per:children": 28, "org:parents": 29, "per:cities_of_residence": 30, "per:stateorprovince_of_birth": 31, "per:charges": 32, "org:founded": 33, "org:founded_by": 34, "per:stateorprovince_of_death": 35, "org:members": 36, "per:country_of_birth": 37, "per:alternate_names": 38, "org:number_of_employees/members": 39, "org:dissolved": 40, "org:political/religious_affiliation": 41}
ace05_reltoid = {"NONE":0,"PHYS":1,"GEN-AFF":2,"PER-SOC":3,"ORG-AFF":4,"PART-WHOLE":5,"ART":6}
ace05_idtoprompt = {0:"NONE",1:"PHYSICAL",2:"GENERAL AND AFFILIATION",3:"PERSON AND SOCIAL",4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}

semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}

semeval_idtoprompt = {0:"NONE",1:"CAUSE AND EFFECT", 2:"COMPONENT AND WHOLE", 3:"ENTITY AND DESTINATION",4:"ENTITY AND ORIGIN",5:"PRODUCT AND PRODUCER",6:"MEMBER AND COLLECTION",7:"MESSAGE AND TOPIC",8:"CONTENT AND CONTAINER",9:"INSTRUMENT AND AGENCY"}

scierc_reltoid = {"NONE": 0, "PART-OF": 1, "USED-FOR": 2, "FEATURE-OF": 3, "CONJUNCTION": 4, "EVALUATE-FOR": 5, "HYPONYM-OF": 6, "COMPARE": 7}

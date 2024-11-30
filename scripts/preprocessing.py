import csv
import pandas as pd
import typing

# 2: transaction_reference_id,party_role,party_info_unstructured,
# 8: parsed_name,parsed_address_street_name,
# parsed_address_street_number,parsed_address_unit,parsed_address_postal_code,parsed_address_city,
# parsed_address_state,parsed_address_country,
# 3: party_iban,party_phone,external_id
class Party:
    tid: str = ""
    role: str = ""
    iban: str = ""
    info_unstructured: str = ""
    phone: str = ""

    pname: str = ""
    paddress_street_name: str = ""
    paddress_street_number: str = ""
    paddress_street_unit: str = ""
    paddress_street_postal_code: str = ""
    paddress_city: str = ""
    paddress_state: str = ""
    paddress_country: str = ""

    # A label assigned to each external party in the training dataset. 
    # This label groups external parties that refer to the same underlying entity
    # (i.e., they represent the same physical or moral entity). 
    # This field is only available in the training dataset and is the target
    # for your model’s predictions
    eid: str = ""

    def __str__(self):  
        txt = ""
        txt += f"TID: {self.tid}\n"
        txt += f"ROLE: {self.role}\n"
        txt += f"IBAN: {self.iban}\n"
        txt += f"INFO: {self.info_unstructured}\n"
        txt += f"PHONE: {self.phone} -> {self.format_phone()}\n"
        txt += f"EID: {self.eid}\n"
        txt += "=" * 20
        return txt

    def format_phone_aux(self, num: str):
        res = "".join([ele for ele in num if ele.isdigit()])
        return res.strip("0")

    def format_phone(self):
        ss = self.phone.split(")")
        res = ""
        for s in ss:
            res += self.format_phone_aux(s)
        return res


    def to_dict(self):
        return {
            "tid": self.tid,
            "role": self.role,
            "iban": self.iban,
            "info": self.info_unstructured,

            "name": self.pname,
            "addr_street_name": self.paddress_street_name,
            "addr_street_number": self.paddress_street_number,
            "addr_street_unit": self.paddress_street_unit,
            "addr_street_postal_code": self.paddress_street_postal_code,
            "addr_city": self.paddress_city,
            "addr_state": self.paddress_state,
            "addr_country": self.paddress_country,

            "phone": self.format_phone(),
            "eid": self.eid
        }

    def to_pd(self):
        return pd.DataFrame(self.to_dict(), index=[0])

class Entity:
    def __init__(self):
        self.parties: typing.List[Party] = []

    def add_party(self, party: Party):
        self.parties.append(party)

    def __str__(self):  
        txt = ""
        for p in self.parties:
            txt += p.__str__() + "\n"
            txt += "=" * 20 + "\n"
        print(len(self.parties))
        txt += "-" * 30 + "\n"
        return txt

"""
group by same info field
assumption: a party is an entity
"""
def byinfo(data: typing.List[Entity]) -> typing.List[Entity]:

    group: typing.Dict[str, typing.List[Entity]] = {}

    print(len(data))

    # group by phone 
    for bigent in data:
        for p in bigent.parties:
            criteria = p.info_unstructured
            if criteria in group:
                # put the whole entity
                if bigent not in group[criteria]:
                    group[criteria].append(bigent)
            else:
                group[criteria] = [bigent]


    print(len(group))
    count = 0
    tot = 0 
    entities: typing.List[Entity] = []
    # merge by group
    for i in group:
        bigent: Entity = Entity()
        for ent in group[i]:
            bigent.parties += ent.parties

        entities.append(bigent)

        if len(group[i]) > 1:
            count +=1 
            tot += len(group[i])

    print(f"by info\n\t{tot} -> {count}\n\t{len(data)} -> {len(entities)}")

    assert len(data) >= len(entities)
    return entities

"""
group by same phone field
assumption: a party is an entity
"""
def byphone(data: typing.List[Entity]) -> typing.List[Entity]:

    group: typing.Dict[str, typing.List[Entity]] = {}

    print(len(data))

    # group by phone 
    for bigent in data:
        for p in bigent.parties:
            phone = p.format_phone()
            if phone in group:
                # put the whole entity
                if bigent not in group[phone]:
                    group[phone].append(bigent)
            else:
                group[phone] = [bigent]


    print(len(group))
    count = 0
    tot = 0 
    entities: typing.List[Entity] = []
    # merge by group
    for i in group:
        bigent: Entity = Entity()
        for ent in group[i]:
            bigent.parties += ent.parties

        entities.append(bigent)

        if len(group[i]) > 1:
            count +=1 
            tot += len(group[i])

    print(f"by phone\n\t{tot} -> {count}\n\t{len(data)} -> {len(entities)}")

    assert len(data) >= len(entities)
    return entities

"""
group by same iban field
assumption: a party is an entity
"""
def byiban(data: typing.List[Entity]) -> typing.List[Entity]:

    group: typing.Dict[str, typing.List[Entity]] = {}

    # group by info 
    for bigent in data:
        for p in bigent.parties:
            criteria = p.iban
            if criteria in group:
                # put the whole entity
                if bigent not in group[criteria]:
                    group[criteria].append(bigent)
            else:
                group[criteria] = [bigent]


    print("iban groups", len(group))
    count = 0
    tot = 0 
    entities: typing.List[Entity] = []
    revmap: typing.Dict[Entity, Entity] = {}

    # merge by group
    for i in group:
        bigent: Entity = Entity()
        collect: None | Entity = None
        should_add = False
        # collect 
        for ent in group[i]:
            if ent in revmap:
                collect = revmap[ent]
            bigent.parties += ent.parties

        if collect == None:
            collect = bigent
            should_add = True
        for ent in group[i]:
            if ent not in revmap:
                collect.parties += ent.parties
                revmap[ent] = collect

            # for party in ent.parties:
            #     bigent.add_party(party)

        if should_add:
            entities.append(collect)

        if len(group[i]) > 1:
            count +=1 
            tot += len(group[i])

    print(f"by iban\n\t{tot} -> {count}\n\t{len(data)} -> {len(entities)}")

    assert len(data) >= len(entities)
    return entities


def group_by_eid(eid_to_party):
    ss = 0
    count = 0
    for ep in eid_to_party:
        ss += len(eid_to_party[ep]) 
        count += 1
        if len(eid_to_party[ep]) > 1:
            print(ep)
            for p in eid_to_party[ep]:
                for pp in p.parties:
                    print(pp.__str__())
            print(len(eid_to_party[ep]))
            print("-"*30 + "\n")

    print(ss/count)



def parse_external_entity(line) -> Entity:

    party: Party = Party()
    party.tid = line[0]
    party.role = line[1]
    party.info_unstructured= line[2]
    party.iban = line[11]
    party.phone= line[12]
    party.eid= line[13]

    entity: Entity = Entity()
    entity.add_party(party)
    return entity 

def print_list(entities: typing.List[Entity]):
    for l in entities:
        print(l)
            

def main():

    ept = open("./raw/external_parties_train.csv", "r")
    abt = open("./raw/account_booking_train.csv", "r")
    eptr = csv.reader(ept, delimiter=',')
    abtr = csv.reader(abt, delimiter=',')

    eids = {}
    aids = {}

    entities:typing.Dict[str, Entity]  = {}

    eid_to_party = {}

    twolegs = []
    orphans = []
    sus = []
   
    first =True 
    for line in eptr:
        if first:
            first = False
            continue
        id = line[0]
        eids[id] = True
        entities[id] = parse_external_entity(line)


    # two legs
    first =True 
    for line in abtr:
        if first:
            first = False
            continue
        id = line[0]
        if id in aids:
            twolegs.append(id)
            if id in entities:
                del entities[id]

        aids[id] = True
        if id not in eids:
            orphans.append(id)

    # is it sound ?
    for line in eptr:
        id = line[0]
        if id not in aids:
            sus.append(id)

    entities_list = []
    for id in entities:
        elem = entities[id]
        entities_list.append(elem)
        if elem.parties[0].eid not in eid_to_party:
            eid_to_party[elem.parties[0].eid] = [elem]
        else:
            eid_to_party[elem.parties[0].eid].append(elem)

            # print(line)
    # print(len(twolegs))
    # print(len(orphans))
    assert len(twolegs) == len(orphans) / 2

    ept.close()
    abt.close()

    # group_by_eid(eid_to_party)

    print(len(entities_list))
    # print(len(entities_list[0].parties))
    entities_list = byinfo(entities_list);
    print(len(entities_list))
    print("="* 40)
    entities_list = byphone(entities_list);
    print(len(entities_list))
    print("="* 40)
    entities_list = byiban(entities_list);
    print(len(entities_list))
    print("="* 40)

    print_list(entities_list);

main()

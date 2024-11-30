import csv
import pandas as pd
import typing
from enum import Enum
from datasketch import MinHash
from datasketch import MinHashLSH

from lexrank import find_rank



class Criteria(Enum):
    IBAN = 1
    PHONE = 2
    INFO_EXACT = 3
    LEXRANK = 4
    NAME_EXACT_AND_STREET_NAME = 5

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
    pname_rank: int = 0
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
        txt += f"NAME: {self.pname} - {self.pname_rank}\n"
        txt += f"STREET NAME: {self.paddress_street_name}\n"
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
        txt += (f"N. parties: {len(self.parties)}\n")
        txt += "-" * 30 + "\n"
        return txt

def getcriteria(ent: Party, criteria: Criteria) -> str | None:
    match criteria:
        case Criteria.IBAN:
            if ent.iban == "":
                return None
            return ent.iban
        case Criteria.PHONE:
            if ent.phone == "":
                return None
            return ent.format_phone()
        case Criteria.INFO_EXACT:
            if ent.info_unstructured == "":
                return None
            return ent.info_unstructured
        case Criteria.LEXRANK:
            if ent.pname == "":
                return None
            return str(ent.pname_rank)
        case Criteria.NAME_EXACT_AND_STREET_NAME:
            if ent.pname == "" or ent.paddress_street_name == "":
                return None
            return (ent.pname + ent.paddress_street_name).replace(" ", "")
        case _:
            return None



def bycriteria(data: typing.List[Entity], crit: Criteria) -> typing.List[Entity]:
    group: typing.Dict[str, typing.List[Entity]] = {}

    # group by info
    for bigent in data:
        for p in bigent.parties:
            criteria: str | None = getcriteria(p, crit)
            if criteria is not None:
                if criteria in group:
                    # put the whole entity
                    if bigent not in group[criteria]:
                        group[criteria].append(bigent)
                else:
                    group[criteria] = [bigent]
                if len(group[criteria]) > 6:
                    for e in group[criteria]:
                        print(e)
                    print("~" * 40 + "\n\n")
                    raise Exception(
                        f"Ohhhhhh too much 1: {len(group[criteria])}")
            else:
                # print(f"WARNING: Listen girl not None: {crit}")
                pass

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
                if should_add == False:
                    collect.parties += ent.parties
                revmap[ent] = collect

            # for party in ent.parties:
            #     bigent.add_party(party)

        if should_add:
            # print(collect)
            entities.append(collect)

        if len(group[i]) > 1:
            count += 1
            tot += len(group[i])
            if len(group[i]) > 6:
                for e in group[i]:
                    print(e)
                print("~" * 40 + "\n\n")
                raise Exception(f"Ohhhhhh too much 2: {len(group[i])}")

    for d in data:
        if d not in revmap:
            entities.append(d)

    print(f"by {crit}\n\t{tot} -> {count}\n\t{len(data)} -> {len(entities)}")

    assert len(data) >= len(entities)

    print(len(entities))
    print("=" * 40)

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


# 2: transaction_reference_id,party_role,party_info_unstructured,
# 8: parsed_name,parsed_address_street_name,
# parsed_address_street_number,parsed_address_unit,parsed_address_postal_code,parsed_address_city,
# parsed_address_state,parsed_address_country,
# 3: party_iban,party_phone,external_id
def parse_external_entity(line) -> Entity:

    party: Party = Party()
    party.tid = line[0]
    party.role = line[1]
    party.info_unstructured = line[2]
    party.pname = line[3]
    party.pname_rank = find_rank(party.pname)

    party.paddress_street_name = line[4]
    party.paddress_street_number = line[5]

    party.iban = line[11]
    party.phone = line[12]
    party.eid = line[13]

    entity: Entity = Entity()
    entity.add_party(party)
    return entity


def print_list(entities: typing.List[Entity]):
    for l in entities:
        if len(l.parties) > 2:
            print(l)


def count_parties(entities: typing.List[Entity]) -> int:
    s: int = 0
    for l in entities:
        s += len(l.parties)
    return s


def verify(entities: typing.List[Entity]) -> bool:
    for l in entities:
        eid = None
        for p in l.parties:
            if eid == None:
                eid = p.eid
            elif eid != p.eid:
                print(f"Error girll -> {eid} != {p.eid}")
                print(l)
                return False

    return True


def main():

    ept = open("./raw/external_parties_train.csv", "r")
    abt = open("./raw/account_booking_train.csv", "r")
    eptr = csv.reader(ept, delimiter=',')
    abtr = csv.reader(abt, delimiter=',')

    eids = {}
    aids = {}

    entities: typing.Dict[str, Entity] = {}

    eid_to_party = {}

    twolegs = []
    orphans = []
    sus = []

    first = True
    for line in eptr:
        if first:
            first = False
            continue
        id = line[0]
        eids[id] = True
        entities[id] = parse_external_entity(line)

    # two legs
    first = True
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

    n_parties = count_parties(entities_list)

    print(len(entities_list))
    
    entities_list = bycriteria(entities_list, Criteria.PHONE)
    entities_list = bycriteria(entities_list, Criteria.IBAN)
    entities_list = bycriteria(entities_list, Criteria.INFO_EXACT)
    # entities_list = bycriteria(entities_list, Criteria.LEXRANK)
    entities_list = bycriteria(entities_list, Criteria.NAME_EXACT_AND_STREET_NAME)
    # print_list(entities_list);

    n_parties_new = count_parties(entities_list)
    print(f"{n_parties} -> {n_parties_new}")

    assert n_parties == n_parties_new
    assert verify(entities_list)


main()

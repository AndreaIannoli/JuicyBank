import csv
from tokenize import group
import pandas as pd
import typing
import numpy as np
from enum import Enum
from collections import Counter
import jellyfish
from fuzzywuzzy import fuzz
import pickle

# from lsh import lsh
from lsh_new import lsh

allwords = []
lshsets = {}

class Criteria(Enum):
    IBAN = 1
    PHONE = 2
    INFO_EXACT = 3
    DUMB_RANKING = 4
    NAME_EXACT_AND_STREET_NAME = 5
    EVIL_CORP = 6
    ADDRESS = 7
    NAME_MOD = 8
    NAME_EXACT_AND_STREET_CODE = 9
    NAME_EXACT_AND_CITY = 10
    LSH_SECRET = 11

# 2: transaction_reference_id,party_role,party_info_unstructured,
# 8: parsed_name,parsed_address_street_name,
# parsed_address_street_number,parsed_address_unit,parsed_address_postal_code,parsed_address_city,
# parsed_address_state,parsed_address_country,
# 3: party_iban,party_phone,external_id


def dumb_ranking(name: str) -> str:
    rank = 0
    i = 0
    for s in name.replace(" ", ""):
        i += 1
        rank += i+ord(s)
    return str(len(name)) + str(rank)

class Party:
    tid: str = ""
    role: str = ""
    iban: str = ""
    info_unstructured: str = ""
    phone: str = ""

    pname: str = ""
    pname_mod: str = ""
    paddress_street_name: str = ""
    paddress_street_number: str = ""
    paddress_street_unit: str = ""
    paddress_street_postal_code: str = ""
    paddress_city: str = ""
    paddress_city_mod: str = ""
    paddress_state: str = ""
    paddress_country: str = ""

    isevilcorp: bool = False
    ranking: str = ""

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
        txt += f"NAME: {self.pname} - '{self.ranking}'\n"
        txt += f"NAME MOD: {self.pname_mod}\n"
        txt += f"STREET NAME: {self.paddress_street_name}\n"
        txt += f"STREET NUM: {self.paddress_street_number}\n"
        txt += f"STREET UNIT: {self.paddress_street_unit}\n"
        txt += f"STREET CODE: {self.paddress_street_postal_code}\n"
        txt += f"CITY: {self.paddress_city}\n"
        txt += f"STATE: {self.paddress_state}\n"
        txt += f"COUNTRY: {self.paddress_country}\n"
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

def isevilcorp(name: str) -> bool:
    evilcorpnouns = ["corp", "inc", "ltd", "firm"]
    return any(substring in name for substring in evilcorpnouns)

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
            return ent.info_unstructured.replace(" ", "")
        case Criteria.DUMB_RANKING:
            if ent.ranking == "":
                return None
            return ent.ranking
        case Criteria.NAME_EXACT_AND_STREET_NAME:
            if ent.pname_mod == "" or ent.paddress_street_name == "":
                return None
            return (ent.pname_mod + ent.paddress_street_name).replace(" ", "")
        case Criteria.NAME_EXACT_AND_STREET_CODE:
            if ent.pname_mod == "" or ent.paddress_street_postal_code== "":
                return None
            return (ent.pname_mod + ent.paddress_street_postal_code).replace(" ", "")
        case Criteria.EVIL_CORP:
            if ent.pname_mod == "" or not ent.isevilcorp:
                return None
            return ent.pname_mod.replace(" ", "")
        case Criteria.ADDRESS:
            if ent.paddress_street_name == "" or ent.paddress_street_number == "":
                return None
            return (ent.paddress_street_name + ent.paddress_street_number).replace(" ", "")
        case Criteria.NAME_MOD:
            if ent.pname_mod == "" or not ent.isevilcorp or ent.paddress_street_postal_code == "":
                return None
            return (ent.pname_mod + ent.paddress_street_postal_code).replace(" ", "")
        case Criteria.NAME_EXACT_AND_CITY:
            if ent.pname_mod == "" or ent.paddress_city_mod == "":
                return None
            return (ent.pname_mod + ent.paddress_city_mod).replace(" ", "")
        case Criteria.LSH_SECRET:
            # nn = ent.pname.replace(" ", "")
            # if ent.pname == "" or nn not in lshsets or ent.paddress_street_name == "":
            #     return None
            # new_name = ""
            # new_name += str(lshsets[nn])
            # return (new_name+ ent.paddress_street_name).replace(" ", "")
            nn = ent.pname.replace(" ", "")
            if ent.pname == "" or nn not in lshsets:
                return None
            new_name = ""
            new_name += str(lshsets[nn])
            return new_name
        case _:
            return None

def comparebycity(p1: Party, p2: Party) -> bool:
    ratio_name = p1.pname != "" and p2.pname != "" and fuzz.token_sort_ratio(p1.pname, p2.pname) > 80
    tt = p1.paddress_street_number != "" and p2.paddress_street_number != "" and p1.paddress_street_number == p2.paddress_street_number and ratio_name
    if tt:
        return True
    tt = p1.paddress_street_postal_code != "" and p2.paddress_street_postal_code != "" and p1.paddress_street_postal_code== p2.paddress_street_postal_code and ratio_name
    return tt

def comparebystate(p1: Party, p2: Party) -> bool:
    ratio_name = p1.pname != "" and p2.pname != "" and fuzz.token_sort_ratio(p1.pname, p2.pname) > 80
    tt = p1.paddress_street_number != "" and p2.paddress_street_number != "" and p1.paddress_street_number == p2.paddress_street_number and ratio_name
    if tt:
        return True
    tt = p1.paddress_street_postal_code != "" and p2.paddress_street_postal_code != "" and p1.paddress_street_postal_code== p2.paddress_street_postal_code and ratio_name
    return tt


# try to add an alone entity to a group
def findfriendsbycity(entities: typing.List[Entity]) -> typing.List[Entity]:

    group_ent: typing.List[Entity] = []
    alone_ent: typing.List[Entity] = []
    for l in entities:
        if len(l.parties) == 1:
            alone_ent.append(l)
        else:
            group_ent.append(l)

    citymap: typing.Dict[str, typing.List[Entity]] = {}
    # for ent in entities:
    for ent in group_ent:
        for p in ent.parties:
            if p.paddress_city != "":
                if p.paddress_city not in citymap:
                    citymap[p.paddress_city] = [ent]
                elif ent not in citymap[p.paddress_city]:
                    citymap[p.paddress_city].append(ent)

    to_skip = {}
    for l in alone_ent:
        if l in to_skip:
            continue
        # if len(l.parties) == 1:
        if l.parties[0].pname != "" and l.parties[0].paddress_city != "" and l.parties[0].paddress_city in citymap:
            gg = citymap[l.parties[0].paddress_city]
            stop = False
            for n in gg:
                if n in to_skip:
                    continue
                for p in n.parties:
                    if l.parties[0].tid != p.tid and comparebycity(l.parties[0], p):
                        n.parties += l.parties[:]
                        alone_ent.remove(l)
                        # to_skip[l] = True
                        # to_skip[n] = True
                        stop = True
                        break
                if stop == True:
                    break

    return alone_ent + group_ent

def findfriendsbystate(entities: typing.List[Entity]) -> typing.List[Entity]:

    group_ent: typing.List[Entity] = []
    alone_ent: typing.List[Entity] = []
    for l in entities:
        if len(l.parties) == 1:
            alone_ent.append(l)
        else:
            group_ent.append(l)

    statemap: typing.Dict[str, typing.List[Entity]] = {}
    # for ent in entities:
    for ent in group_ent:
        for p in ent.parties:
            if p.paddress_state!= "":
                if p.paddress_state not in statemap:
                    statemap[p.paddress_state] = [ent]
                elif ent not in statemap[p.paddress_state]:
                    statemap[p.paddress_state].append(ent)

    # print(statemap.keys())
    if "marshallislands" in statemap:
        print(len(statemap["marshallislands"]))
    to_skip = {}
    for l in alone_ent:
        if l in to_skip:
            continue
        # if len(l.parties) == 1:
        if l.parties[0].pname != "" and l.parties[0].paddress_state != "" and l.parties[0].paddress_state in statemap:
            gg = statemap[l.parties[0].paddress_state]
            stop = False
            for n in gg:
                if n in to_skip:
                    continue
                for p in n.parties:
                    if l.parties[0].tid != p.tid and comparebystate(l.parties[0], p):
                        n.parties += l.parties[:]
                        alone_ent.remove(l)
                        # to_skip[l] = True
                        # to_skip[n] = True
                        stop = True
                        break
                if stop == True:
                    break

    return alone_ent + group_ent

def postprocessnames(entities: typing.List[Entity]) -> typing.List[Entity]:

    for l in entities:
        for p in l.parties:
            if p.pname != "" and p.pname in lshsets:
                p.pname_mod = str(lshsets[p.pname])
            else:
                p.pname_mod = p.pname
            if p.paddress_city != "" and p.paddress_city in lshsets:
                p.paddress_city_mod = str(lshsets[p.paddress_city])
            else:
                p.paddress_city_mod = p.paddress_city

    return entities


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
                # if len(group[criteria]) > 6:
                #     for e in group[criteria]:
                #         print(e)
                #     print("~" * 40 + "\n\n")
                #     raise Exception(
                #         f"Ohhhhhh too much 1: {len(group[criteria])}")
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
            # if len(group[i]) > 6:
            #     for e in group[i]:
            #         print(e)
            #     print("~" * 40 + "\n\n")
            #     raise Exception(f"Ohhhhhh too much 2: {len(group[i])}")

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


def remove_duplicates(input: str) -> str:
 
    # split input string separated by space
    ss = input.split(" ")
 
    # now create dictionary using counter method
    # which will have strings as key and their 
    # frequencies as value
    UniqW = Counter(ss)
 
    # joins two adjacent elements in iterable way
    s = " ".join(UniqW.keys())
    return s


# 2: transaction_reference_id,party_role,party_info_unstructured,
# 8: parsed_name,parsed_address_street_name,
# parsed_address_street_number,parsed_address_unit,parsed_address_postal_code,parsed_address_city,
# parsed_address_state,parsed_address_country,
# 3: party_iban,party_phone,external_id
def parse_external_entity(line) -> Entity:

    party: Party = Party()
    party.tid = line[0].lower()
    party.role = line[1].lower()
    party.info_unstructured = line[2].lower()
    party.pname = line[3].lower()

    party.paddress_street_name = line[4].lower().replace(" ", "")
    party.paddress_street_number = line[5].lower().replace(" ", "")
    party.paddress_street_unit = line[6].lower().replace(" ", "")
    party.paddress_street_postal_code = line[7].lower().replace(" ", "")

    party.paddress_city = line[8].lower().replace(" ", "")
    if party.paddress_city != "":
        allwords.append(party.paddress_city.replace(" ", ""))


    party.paddress_state = line[9].lower().replace(" ", "")
    party.paddress_country = line[10].lower().replace(" ", "")

    party.iban = line[11].lower()
    party.phone = line[12].lower()
    if len(line) >= 14:
        party.eid = line[13]


    if party.pname != "":
        party.pname = party.pname.replace('mr. ', '')
        party.pname = party.pname.replace('ms. ', '')
        party.pname = party.pname.replace('mrs. ', '')
        party.pname = party.pname.replace('miss ', '')
        party.pname = party.pname.replace('dr. ', '')
        party.pname = party.pname.replace('prof. ', '')
        party.pname = party.pname.replace('rev. ', '')
        party.pname = party.pname.replace('hon. ', '')


        party.isevilcorp = isevilcorp(party.pname)
        allwords.append(party.pname.replace(" ", ""))

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

def show_singleton(entities: typing.List[Entity], eid_list: typing.Dict[str, typing.List[Entity]]):
    alone = 0
    for l in entities:
        check = l.parties[0].paddress_city != "" and (l.parties[0].paddress_street_name != "" or l.parties[0].paddress_street_number != "" or  l.parties[0].paddress_street_unit != "" or  l.parties[0].paddress_street_postal_code != "" )
        if len(l.parties) == 1 and len(eid_list[l.parties[0].eid]) > 1 and check:
            print(l)
            print("?"* 40 + "\n")
            for e in eid_list[l.parties[0].eid]:
                if e != l.parties[0]:
                    print(e.__str__().replace("\n", "\n\t"))
            print("|"* 40 + "\n")
        if len(l.parties) == 1:
            alone +=1
    print(f"Alone: {alone}")

def remove_singleton(entities: typing.List[Entity]):
    new_entities: typing.List[Entity] = []
    for l in entities:
        if len(l.parties) != 1:
            new_entities.append(l)
    return new_entities


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

    istest = False
    ept = {}
    abt = {}
    if istest :
        ept = open("./raw/external_parties_test.csv", "r")
        abt = open("./raw/account_booking_test.csv", "r")
    else:
        ept = open("./raw/external_parties_train.csv", "r")
        abt = open("./raw/account_booking_train.csv", "r")
    eptr = csv.reader(ept, delimiter=',')
    abtr = csv.reader(abt, delimiter=',')

    eids = {}
    aids = {}

    entities: typing.Dict[str, Entity] = {}

    eid_to_party: typing.Dict[str, typing.List[Entity]] = {}

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


    # result = lsh.query(minhash)
    # nn = np.array(result)
    # print(nn.shape)


    n_parties = count_parties(entities_list)

    global lshsets
    # lshsets = lsh(allwords)
    # print(lshsets)
    # with open('file.pkl', 'rb') as file: 
    #     lshsets = pickle.load(file)
    print("lsh computed")

    # entities_list = postprocessnames(entities_list)

    entities_list = bycriteria(entities_list, Criteria.PHONE)
    entities_list = bycriteria(entities_list, Criteria.IBAN)
    entities_list = bycriteria(entities_list, Criteria.INFO_EXACT)
    entities_list = bycriteria(entities_list, Criteria.NAME_EXACT_AND_STREET_NAME)
    entities_list = bycriteria(entities_list, Criteria.ADDRESS)
    entities_list = bycriteria(entities_list, Criteria.NAME_EXACT_AND_CITY)
    entities_list = bycriteria(entities_list, Criteria.NAME_EXACT_AND_STREET_CODE)
    # print_list(entities_list);

    # group_by_eid(eid_to_party)

    entities_list = findfriendsbycity(entities_list)
    entities_list = findfriendsbycity(entities_list)
    entities_list = findfriendsbystate(entities_list)
    entities_list = findfriendsbystate(entities_list)
    # entities_list = findfriends(entities_list)
    # entities_list = findfriends(entities_list)

    # print(lshsets)

    # print(lshsets['joshualopez'])
    # print(lshsets['joshualopez'])


    # show_singleton(entities_list, eid_to_party)

    # print(len(entities_list))

    # entities_list = remove_singleton(entities_list)

    n_parties_new = count_parties(entities_list)
    print(f"{n_parties} -> {n_parties_new}")
    print(len(entities_list))

    # with open('file.pkl', 'wb') as file:
    #     pickle.dump(lshsets, file)

    assert n_parties == n_parties_new
    if not istest:
        assert verify(entities_list)

    print("passed")


main()

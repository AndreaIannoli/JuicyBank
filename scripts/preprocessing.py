import csv
import pandas as pd
import typing

# 2: transaction_reference_id,party_role,party_info_unstructured,
# 8: parsed_name,parsed_address_street_name,
# parsed_address_street_number,parsed_address_unit,parsed_address_postal_code,parsed_address_city,
# parsed_address_state,parsed_address_country,
# 3: party_iban,party_phone,external_id
class ExternalEntity:
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
        txt += f"PHONE: {self.phone}\n"
        txt += f"EID: {self.eid}\n"
        txt += "=" * 20
        return txt

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

            "phone": self.phone,
            "eid": self.eid
        }

    def to_pd(self):
        return pd.DataFrame(self.to_dict(), index=[0])

def herot():

    pass



def parse_external_entity(line) -> ExternalEntity:

    entity: ExternalEntity = ExternalEntity()
    entity.tid = line[0]
    entity.role = line[1]
    entity.info_unstructured= line[2]
    entity.iban = line[11]
    entity.phone= line[12]
    entity.eid= line[13]

    return entity
            

def main():

    ept = open("./raw/external_parties_train.csv", "r")
    abt = open("./raw/account_booking_train.csv", "r")
    eptr = csv.reader(ept, delimiter=',')
    abtr = csv.reader(abt, delimiter=',')

    eids = {}
    aids = {}

    external:typing.Dict[str, ExternalEntity]  = {}

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
        external[id] = parse_external_entity(line)
        # print(external[id].to_pd())


    # two legs
    first =True 
    for line in abtr:
        if first:
            first = False
            continue
        id = line[0]
        if id in aids:
            twolegs.append(id)
            if id in external:
                del external[id]

        aids[id] = True
        if id not in eids:
            orphans.append(id)

    # is it sound ?
    for line in eptr:
        id = line[0]
        if id not in aids:
            sus.append(id)

    for id in external:
        elem = external[id]
        if elem.eid not in eid_to_party:
            eid_to_party[elem.eid] = [elem]
        else:
            eid_to_party[elem.eid].append(elem)

            # print(line)
    print(len(twolegs))
    print(len(orphans))
    assert len(twolegs) == len(orphans) / 2

    ept.close()
    abt.close()

    ss = 0
    count = 0
    for ep in eid_to_party:
        ss += len(eid_to_party[ep]) 
        count += 1
        if len(eid_to_party[ep]) > 1:
            print(ep)
            for p in eid_to_party[ep]:
                print(p.__str__())
            print(len(eid_to_party[ep]))
            print("-"*30 + "\n")

    print(ss/count)

main()

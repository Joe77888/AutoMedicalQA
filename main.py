from py2neo import Graph
from train_model import IntentSlotModel
import torch
from transformers import AutoTokenizer
from config import *
from collections import defaultdict
import warnings


warnings.filterwarnings('ignore')
template_dict = {
    'disease_symptom': 'MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name in [%s] return m.name, n.name',
    'symptom_disease': 'MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where n.name in [%s] return m.name, n.name',
    'disease_cause': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.cause',
    'disease_accompany': 'MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name in [%s] return m.name, n.name',
    'disease_not_food': 'MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name in [%s] return m.name, n.name',
    'disease_food': 'MATCH (m:Disease)-[r:do_eat|recommand_eat]->(n:Food) where m.name in [%s] return m.name, n.name',
    'food_not_disease': 'MATCH (m:Disease)-[r:no_eat]->(n:Food) where n.name in [%s] return m.name, n.name',
    'food_disease': 'MATCH (m:Disease)-[r:do_eat|recommand_eat]->(n:Food) where n.name in [%s] return m.name, n.name',
    'disease_drug': 'MATCH (m:Disease)-[r:common_drug|recommand_drug]->(n:Drug) where m.name in [%s] return m.name,  n.name',
    'drug_disease': 'MATCH (m:Disease)-[r:common_drug|recommand_drug]->(n:Drug) where n.name in [%s] return m.name, n.name',
    'disease_check': 'MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name in [%s] return m.name, n.name',
    'check_disease': 'MATCH (m:Disease)-[r:need_check]->(n:Check) where n.name in [%s] return m.name, n.name',
    'disease_prevent': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.prevent',
    'disease_duration': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.cure_lasttime',
    'disease_cure_prob': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.cured_prob',
    'disease_cure_method': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.cure_way',
    'disease_people': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.easy_get',
    'disease_desc': 'MATCH (m:Disease) where m.name in [%s] return m.name, m.desc',
    'others': None
}

answer_dict = {'disease_symptom': '{0}??????????????????{1}',
               'symptom_disease': '{0}???????????????????????????{1}',
               'disease_cause': '{0}?????????????????????{1}',
               'disease_prevent': '{0}????????????????????????{1}',
               'disease_duration': '{0}???????????????????????????{1}',
               'disease_cure_method': '{0}???????????????????????????{1}',
               'disease_cure_prob': '{0}???????????????????????????????????????{1}',
               'disease_people': '{0}????????????????????????{1}',
               'disease_desc': '{0}????????????{1}',
               'disease_accompany': '{0}??????????????????{1}',
               'disease_not_food': '???{0}???????????????????????????{1}',
               'disease_food': '???{0}???????????????????????????{1}',
               'food_not_disease': '??????{0}?????????????????????{1}',
               'food_disease': '??????{0}?????????????????????{1}',
               'disease_drug': '??????{0}?????????????????????????????????{1}',
               'drug_disease': '{0}??????????????????{1}',
               'disease_check': '{0}?????????????????????????????????????????????{1}',
               'check_disease': '????????????{0}????????????????????????{1}'
               }


class Text2Cypher:
    def __init__(self):
        print('????????????????????????????????????, ????????????...')
        self.graph = Graph("http://localhost:7474", auth=(username, password))
        with open('data/labels.txt') as f:
            self.idx_to_intent_dict = {i: v for i, v in enumerate(f.read().split('\n'))}
        self.intent_slot_model = IntentSlotModel('hfl/chinese-roberta-wwm-ext')
        self.intent_slot_model.load_state_dict(torch.load('ckpt/chinese-roberta-wwm-ext.ckpt', map_location=torch.device('cpu')))
        self.intent_slot_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    @staticmethod
    def get_slots(slot_idx):
        slots = []
        left = 0
        while left < len(slot_idx):
            if slot_idx[left] != 1:
                left += 1
            else:
                right = left + 1
                while slot_idx[right] == 2:
                    right += 1
                slots.append([left, right])
                left = right
        return slots

    def get_intent_and_slots(self, query):

        embeddings = self.tokenizer([query])
        intent_output, slot_output = self.intent_slot_model(input_ids=torch.tensor(embeddings['input_ids']),
                                                            attention_mask=torch.tensor(embeddings['attention_mask']))
        intent = self.idx_to_intent_dict[intent_output.argmax(-1).item()]

        slot_idx = slot_output.argmax(-1)[0].numpy()
        slots = self.get_slots(slot_idx)
        slots = [query[i[0]+1:i[1]+1] for i in slots]
        return intent, slots

    def generate_answer(self, intent, slots, res_dict):
        answers = []
        value_list = []
        for i in res_dict.values():
            value_list.extend(i)
        not_found = ', '.join(set(slots) - set(res_dict.keys()) - set(value_list))
        answer_template = answer_dict[intent]
        for k, v in res_dict.items():

            v = v[0] if v[0] is list else v
            v = ', '.join(v)
            answer = answer_template.format(k, v)
            answers.append(answer)
        if not_found:
            answers.append(f'???????????????{not_found}???????????????')
        return '\n'.join(answers)

    def query_res(self, intent, slots):
        query_template = template_dict[intent]
        cypher_query = query_template % ', '.join(['\'' + i + '\'' for i in slots])
        res = self.graph.run(cypher_query).data()
        return res

    def reply(self, query):
        intent, slots = self.get_intent_and_slots(query)
        if intent == 'others':
            reply = '?????????,??????????????????????????????'
        else:
            query_res = self.query_res(intent, slots)

            res_dict = defaultdict(list)

            for res in query_res:
                if type(list(res.values())[1]) == list:
                    res_dict[list(res.values())[0]].extend(list(res.values())[1])
                else:
                    res_dict[list(res.values())[0]].append(list(res.values())[1])
            reply = self.generate_answer(intent, slots, res_dict)
            reply = '?????????,??????????????????????????????' if not reply else reply
        print(reply)


def main():
    text2cypher = Text2Cypher()
    print('- ??????????????????????????????\n - ??????q??????')
    while True:
        print('\n???????????????????????????')
        query = input()
        if query == 'q':
            print('????????????, ???????????????')
            exit()
        else:
            text2cypher.reply(query)


if __name__ == '__main__':
    main()

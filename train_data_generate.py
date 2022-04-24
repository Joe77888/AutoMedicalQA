import random
import json


entity_dict = {
    'disease': [i.strip() for i in open('dict/disease.txt') if i.strip()],
    'check': [i.strip() for i in open('dict/check.txt') if i.strip()],
    'drug': [i.strip() for i in open('dict/drug.txt') if i.strip()],
    'symptom': [i.strip() for i in open('dict/symptom.txt') if i.strip()],
    'food': [i.strip() for i in open('dict/food.txt') if i.strip()]
}

# disease_wds = [i.strip() for i in open('dict/disease.txt') if i.strip()]
# department_wds = [i.strip() for i in open('dict/department.txt') if i.strip()]
# check_wds = [i.strip() for i in open('dict/check.txt') if i.strip()]
# drug_wds = [i.strip() for i in open('dict/drug.txt') if i.strip()]
# food_wds = [i.strip() for i in open('dict/food.txt') if i.strip()]
# producer_wds = [i.strip() for i in open('dict/producer.txt') if i.strip()]
# symptom_wds = [i.strip() for i in open('dict/symptom.txt') if i.strip()]

pattern_dict = {
    'disease_symptom': ['%s有什么症状', '%s的表征有哪些', '得了%s有什么现象', '%s的症状是怎样的', '得了%s有什么表现吗', '患%s会怎样?', '得了%s会怎么样呢'],
    'symptom_disease': ['%s是什么病', '%s是得了什么病', '我%s,可能是什么病', '%s是得了什么病', '%s可能患什么病', '%s是得了感冒吗', '%s可能是得了癌症吗'],
    'disease_cause': ['为什么会%s', '为啥我会得%s', '怎么会得%s', '什么人容易患%s', '怎么样会患%s', '为什么会得%s'],
    'disease_accompany': ['%s有什么并发症', '%s的并发症是什么', '%s的并发症有？', '%s有哪些并发症', '%s有并发症吗'],
    'disease_not_food': ['%s不能吃什么', '得了%s不能吃什么菜', '%s患者有什么忌口', '患%s有哪些忌口', '%s不让吃什么', '吃什么对%s不好', '%s不要吃什么',
                         '得了%s最好不要吃什么', '%s患者最好不要吃什么', '%s有忌口吗'],
    'disease_food': ['%s患者应该吃什么', '得了%s要吃什么', '吃什么能治疗%s', '吃什么菜能治%s', '得了%s要吃点啥', '得%s应该吃啥', '%s患者该吃什么',
                     '吃点什么对治疗%s有帮助', '%s患者的食谱'],
    'food_not_disease': ['什么人最好不要吃%s', '哪些人不应该吃%s', '什么人不能吃%s', '得了什么病最好别吃%s', '什么病人不能吃%s', '患哪种病不要吃%s',
                         '哪些人吃%s不好', '什么人最好别吃%s'],
    'food_disease': ['吃%s有什么好处吗', '哪些人应该吃%s', '什么人应该吃%s', '吃%s能治什么', '吃%s有什么好处', '吃%s能帮助治疗什么呢'],
    'disease_drug': ['%s该吃什么药', '%s应该开什么药', '什么药治%s', '什么药能治%s', '得了%s该开什么药', '我得了%s吃什么药', '%s患者吃什么药呢',
                     '得%s应该吃什么药', '%s该开点啥药啊', '得%s吃什么药'],
    'drug_disease': ['%s是治啥的', '%s治什么病', '%s是治啥的', '%s能治啥', '%s是治疗什么的', '%s可以治什么病', '得什么病要吃%s', '什么病开%s',
                     '%s药是治疗什么病的呢', '%s用于治疗什么', '%s能治啥'],
    'disease_check': ['%s需要做什么检查', '怎么检查%s', '怎么检查有没有得%s', '什么检查可以查出%s', '怎么查%s', '我好像得了%s，该做什么检查',
                      '怀疑自己得了%s该做什么检查', '怎么查有没有患%s', '该怎么查自己得没得%s'],
    'check_disease': ['%s能检查什么病', '什么病可以被%s查出来呢', '%s能查出什么病', '%s是用来查什么病的', '%s可以查哪种病',
                      '%s是用来检查什么病的', '%s可以检查什么病啊'],
    'disease_prevent': ['如何预防%s', '%s怎么才能预防呢', '该怎么预防%s', '不想得%s该怎么办', '怎么防治患%s',
                        '%s该怎么预防', '想要预防%s该怎么办', '不想得%s可以怎么预防', '怎么才能预防%s呢'],
    'disease_duration': ['%s多久能治好', '%s要治疗多久', '%s多久能好', '%s通常多久能治好', '一般情况下%s多久能好', '%s几天能好'],
    'disease_cure_prob': ['%s治好的几率大吗', '%s有多大几率能治好', '治好%s的几率是多少', '%s有多大几率能好', '%s治好的几率有多大', '%s治好的几率大吗',
                          '我得了%s，治好的几率多大', '%s有多大几率治好呀'],
    'disease_cure_method': ['%s要怎么治疗', '怎么治%s', '%s怎么才能治好', '%s要怎么治', '我患了%s怎么治', '如何治疗%s', '想治%s该怎么办',
                            '怎么才能治好%s'],
    'disease_people': ['什么人容易得%s', '哪些人容易得%s', '什么人群会得%s', '%s的易患人群有哪些', '哪些人是%s的易患人群', '%s哪些人容易得'],
    'disease_desc': ['%s相关的信息', '%s是什么病', '%s是什么', '%s是什么病', '描述一下%s', '%s', '%s的科普']}

prefix = ['你好，', '请问一下', '可以告诉我', '我想请问一下', '请问', '我想知道', '告诉我', '你好请问', '请问', '你好', '查询', '科普一下']
connect_words = ['和', ',', '还有', '以及', '，', '、']

all_type = ['disease_symptom', 'symptom_disease', 'disease_cause', 'disease_accompany', 'disease_not_food',
            'disease_food', 'food_not_disease', 'food_disease', 'disease_drug', 'drug_disease', 'disease_check',
            'check_disease', 'disease_prevent', 'disease_duration', 'disease_cure_prob', 'disease_cure_method',
            'disease_people', 'disease_desc']


def data_generate(num):
    random.seed(2022)
    all_data = []
    for n in range(num):
        cur_type = random.choice(all_type)
        cur_prefix = random.choice(prefix)
        cur_entity = cur_type.split('_')[0]

        cur_words = random.sample(entity_dict[cur_entity], random.randint(1, 3))

        cur_words_str = cur_words[0]
        word_idx = [(0, len(cur_words_str))]

        for i in range(len(cur_words) - 1):
            cur_words_str += random.choice(connect_words)
            start = len(cur_words_str)
            cur_words_str += cur_words[i + 1]
            end = len(cur_words_str)
            word_idx.append([start, end])
        # print(cur_words_str, word_idx)

        cur_pattern = random.choice(pattern_dict[cur_type])
        add_lth = len(cur_prefix) + cur_pattern.index('%s')

        query_text = cur_prefix + cur_pattern % cur_words_str
        word_idx = [(i[0] + add_lth, i[1] + add_lth) for i in word_idx]
        all_data.append({'text': query_text, 'class': cur_type, 'idx': word_idx})

    with open('data/zhdd_lines.txt') as f:
        chat_data = [i.split(' ') for i in f.read().split('\n')]
    for i in chat_data:
        for ii in i:
            all_data.append({'text': ii, 'class': 'others', 'idx': []})

    print(len(all_data))
    random.shuffle(all_data)
    all_labels = list(set([i['class'] for i in all_data]))

    train_lth = int(len(all_data) * 0.99)
    with open('data/train_data.json', 'w') as f:
        json.dump(all_data[:train_lth], f, ensure_ascii=False)
    with open('data/test_data.json', 'w') as f:
        json.dump(all_data[train_lth:], f, ensure_ascii=False)
    with open('data/labels.txt', 'w') as f:
        f.write('\n'.join(all_labels))


data_generate(1000000)

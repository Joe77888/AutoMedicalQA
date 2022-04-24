import argparse
import json
import os

import torch.cuda
import transformers
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, RandomSampler, DataLoader
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='data/sample.json')
    parser.add_argument('--test_path', type=str,
                        default='data/sample.json')
    parser.add_argument('--label_path', type=str,
                        default='data/labels.txt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--val_interval', type=int, default=4)

    args = parser.parse_args()
    return vars(args)


class IntentSlotDataset(Dataset):
    def __init__(self, encodings, intents, slots):
        self.input_ids = encodings['input_ids']
        length = len(self.input_ids[0])
        self.attention_mask = encodings['attention_mask']
        self.slots = [i + [100] * (length - len(i)) for i in slots]
        self.intents = intents

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_mask[idx]), \
               torch.tensor(self.intents[idx]), torch.tensor(self.slots[idx])

    def __len__(self):
        return len(self.intents)


class IntentSlotModel(nn.Module):
    def __init__(self, model_name, intent_num=19, slot_num=3):
        super(IntentSlotModel, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.embedding_size = list(self.pretrained_model.named_parameters())[-1][1].size().numel()
        self.intent_dense = nn.Linear(self.embedding_size, intent_num)
        self.slot_dense = nn.Linear(self.embedding_size, slot_num)

    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        intent_output = self.intent_dense(output['pooler_output'])
        slot_output = self.slot_dense(output['last_hidden_state'][:, 1:, :])
        return intent_output, slot_output

    def save_pretrained(self, checkpoint_path):
        print(f'saving model to {checkpoint_path}')
        torch.save(self.state_dict(), checkpoint_path)


def read_data(data_path, label_dict):
    print(f'reading data from {data_path}')
    with open(data_path, 'r') as load_f:
        data = json.load(load_f)
    texts, intents, slots = [], [], []

    # 0: not slot, 1: begin of slot, 2: remaining of slot
    for i in tqdm(data):
        intents.append(label_dict[i['class']])
        texts.append(i['text'])
        slot = [0 for i in range(len(i['text']))]
        for pair in i['idx']:
            slot[pair[0]] = 1
            slot[pair[0] + 1:pair[1]] = [2] * (pair[1] - pair[0] - 1)
        slots.append(slot)
    print(f'num of data: {len(slots)}')
    return texts, intents, slots


def get_label_dict(label_dict_path):
    with open(label_dict_path) as f:
        label_dict = {v: i for i, v in enumerate(f.read().split('\n'))}
    return label_dict


def get_dataloader(data_path, label_dict, tokenizer, batch_size):
    text, intent, slot = read_data(data_path, label_dict)
    text_encoded = tokenizer(text, padding='max_length', max_length=128, truncation=True)
    dataset = IntentSlotDataset(text_encoded, intent, slot)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def train(train_path, test_path, label_path, batch_size, model_name, gpu_id, num_epoch, val_interval):
    transformers.logging.set_verbosity_error()

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f'cuda{gpu_id}'
    else:
        device = 'cpu'
    os.makedirs('ckpt', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_dict = get_label_dict(label_path)
    train_dataloader = get_dataloader(train_path, label_dict, tokenizer, batch_size)
    val_dataloader = get_dataloader(test_path, label_dict, tokenizer, batch_size)

    model = IntentSlotModel(model_name)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    intent_loss_func = nn.CrossEntropyLoss()
    slot_loss_func = nn.CrossEntropyLoss(ignore_index=100)

    best_val_loss = float('inf')

    for epoch in range(num_epoch):
        model.train()
        print(f'start train epoch {epoch}')
        train_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch_input_ids, batch_attention_mask, batch_intent, batch_slot = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_intent = batch_intent.to(device)
            batch_slot = batch_slot.to(device)
            batch_slot = batch_slot[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()
            intent_output, slot_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            slot_output = slot_output.reshape(-1, slot_output.shape[-1])
            intent_loss = intent_loss_func(intent_output, batch_intent)
            slot_loss = slot_loss_func(slot_output, batch_slot)
            loss = intent_loss + slot_loss
            train_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % val_interval == 0 and step != 0:
                val_loss, intent_acc, slot_acc = evaluate(model, val_dataloader, intent_loss_func, slot_loss_func,
                                                          device)
                print(f'val_loss: {val_loss.item()}, intent_acc: {intent_acc}, slot_acc: {slot_acc}')

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_name = f'{model_name}.ckpt'
                    checkpoint_name = checkpoint_name.split('/')[1]
                    model.save_pretrained(f'ckpt/{checkpoint_name}')
                avg_train_loss = train_loss / len(train_dataloader)
                print(f'average train loss: ', avg_train_loss.item())


def evaluate(model, val_dataloader, intent_loss_func, slot_loss_func, device):
    val_loss = 0
    model.eval()
    all_intents, all_slots, all_pred_intents, all_pred_slots = [], [], [], []
    for step, batch in tqdm(enumerate(val_dataloader)):
        batch_input_ids, batch_attention_mask, batch_intent, batch_slot = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_intent = batch_intent.to(device)
        batch_slot = batch_slot.to(device)
        batch_slot = batch_slot[:, 1:].contiguous().view(-1)

        with torch.no_grad():
            intent_output, slot_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        slot_output = slot_output.reshape(-1, slot_output.shape[-1])

        intent_loss = intent_loss_func(intent_output, batch_intent)
        slot_loss = slot_loss_func(slot_output, batch_slot)
        loss = intent_loss + slot_loss
        val_loss += loss

        all_slots.extend(batch_slot.numpy())
        all_intents.extend(batch_intent.numpy())
        all_pred_slots.extend(torch.argmax(slot_output, axis=-1).cpu().numpy())
        all_pred_intents.extend(torch.argmax(intent_output, axis=-1).cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    all_pred_slots = [all_pred_slots[i] for i in range(len(all_pred_slots)) if all_slots[i] != 100]
    all_slots = [i for i in all_slots if i != 100]
    intent_acc = accuracy_score(all_intents, all_pred_intents)
    slot_acc = accuracy_score(all_slots, all_pred_slots)

    return avg_val_loss, intent_acc, slot_acc


if __name__ == '__main__':
    kwargs = get_args()
    train(**kwargs)

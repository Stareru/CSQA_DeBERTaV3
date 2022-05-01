import numpy as np
import torch
from models import CSQAModel, CSQAInference

seed=514
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def read(fname, ftype):
    with open(fname) as fp:
        lines = fp.read().strip().split('\n')

    items = []

    for line in lines:

        evals = eval(line)

        quest = evals['question']['stem']

        choices = [f"{choice['text']}." for choice in evals['question']['choices']]

        quests = [quest for choice in choices]

        if ftype != 'pred':

            label = ['A', 'B', 'C', 'D', 'E'].index(evals['answerKey'])

            items.append({
                'quests': quests,
                'choices': choices,
                'label': label,
            })

        else:

            items.append({
                'quests': quests,
                'choices': choices,
            })

    return items

items_train = read('CSQA/train_rand_split.jsonl', 'train')
items_eval = read('CSQA/dev_rand_split.jsonl', 'eval')
items_pred = read('CSQA/test_rand_split.jsonl', 'pred')

inference = CSQAInference('microsoft/deberta-v3-large', 5)

n_epoch = 4
best = 0.0

for epoch in range(n_epoch):
    inference.train(items_train)
    score = inference.evaluate(items_eval)
    if score > best:
        torch.save(inference.state_dict(), f'./csqa_debertav3.pth')
        best = score
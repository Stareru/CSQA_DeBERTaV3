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

        evals = json.loads(line)

        quest = evals['question']

        if quest[-1] not in ['.', '?']:
            quest += '.'

        choices = ["Yes.", "No."]

        quests = [quest for choice in choices]

        if ftype != 'pred':

            label = ["yes", "no"].index(evals['answer'])

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

items_train = read('CSQA2/teach_your_ai_train.json', 'train')
items_eval = read('CSQA2/teach_your_ai_dev.json', 'eval')

inference = CSQAInference('microsoft/deberta-v3-large', 2)

best = 0.0
save = True

for epoch in range(10):
    inference.train(items_train)
    score = inference.evaluate(items_eval)
    if score > best:
        if save:
            torch.save(inference.state_dict(), f'./csqa2_debertav3.pth')
        best = score
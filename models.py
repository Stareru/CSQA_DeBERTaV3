import torch
from torch import nn
from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CSQAModel(nn.Module):

    def __init__(self, plm, num_label):
        super().__init__()

        self.plm = plm
        self.num_label = num_label

        self.tok = AutoTokenizer.from_pretrained(plm)
        self.model = AutoModelForSequenceClassification.from_pretrained(plm, num_labels=1)

    def forward(self, items):
        '''
        args:

        items [list of dicts] A list of dicts that contain 'choices' [list], 'label' [int]
        '''

        quests = [quest for item in items for quest in item['quests']]
        choices = [choice for item in items for choice in item['choices']]
        inputs = self.tok(quests, choices, padding=True, return_tensors='pt')

        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        scores = self.model(**inputs)['logits'].squeeze(-1)
        scores = scores.reshape(scores.shape[0] // self.num_label, self.num_label)

        return scores


class CSQAInference(nn.Module):

    def __init__(self, plm, num_label):

        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CSQAModel(plm, num_label)
        self.model.device = self.device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam([p for p in self.model.parameters()], lr=1e-5, eps=1e-6, betas=(0.9, 0.999))
        self.scheduler = ExponentialLR(self.optimizer, .67 ** (1 / 5000))

        self.model.to(self.device)

    def _train(self, items):

        self.model.train()

        self.model.zero_grad()

        scores = self.model(items)
        labels = [item['label'] for item in items]
        labels = torch.LongTensor(labels).to(self.device)

        loss = self.criterion(scores, labels)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _evaluate(self, items):

        self.model.eval()

        with torch.no_grad():
            scores = self.model(items)
            preds = scores.argmax(-1).detach().cpu().numpy()
            labels = np.array([item['label'] for item in items])

            return preds, labels

    def _predict(self, items):

        self.model.eval()

        with torch.no_grad():
            scores = self.model(items)
            preds = scores.argmax(-1).detach().cpu().numpy()

            return preds

    def train(self, items, bsize=8):

        np.random.shuffle(items)

        bar = tqdm(range(0, len(items), bsize))

        for idx in bar:
            loss = self._train(items[idx:idx + bsize])

            bar.set_description(f'#Train #Loss:{loss:.3}')

    def evaluate(self, items, bsize=16):

        preds_, labels_ = np.array([]), np.array([])

        bar = tqdm(range(0, len(items), bsize))

        for idx in bar:
            preds, labels = self._evaluate(items[idx:idx + bsize])

            preds_ = np.concatenate([preds_, preds], 0)
            labels_ = np.concatenate([labels_, labels], 0)

            score = self.score(preds_, labels_)

            bar.set_description(f'#Eval #Acc:{score:.3}')

        return self.score(preds_, labels_)

    def predict(self, items, bsize=16):

        preds_ = np.array([])

        bar = tqdm(range(0, len(items), bsize))

        for idx in bar:
            preds = self._predict(items[idx:idx + bsize])

            preds_ = np.concatenate([preds_, preds], 0)

        return predicts

    def score(self, preds, labels):

        return sum(preds == labels) / len(labels)
import random
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from transformers import BertModel, BertTokenizer, AdamW,  get_linear_schedule_with_warmup, set_seed
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
import torch.nn as nn 
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import glob
import os
import pandas as pd


# Зафиксируем все возможные сиды
def seed_all(seed_value):
    random.seed(seed_value) 
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
        
# Использовал для нахождения параметра max_len в токенизаторе
def preprocess_data(df, le):
    df['title'] = df['title'].apply(lambda x: x.lower())
    df = df.dropna()
    libs = pd.DataFrame(np.vstack(df['title'].apply(lambda x: \
        [1 if c in x.split(' ') else 0 for c in le.classes_]).values), columns=le.classes_)
    df = pd.concat((df, libs), axis=1)
    df['sym_len'] = df['title'].apply(len)
    df['word_len'] = df['title'].apply(lambda x: len(x.split()))
    return df
    
   


 # В качестве бейзлайна я брал lemmatization + tfidf + knn, метрика получается ~0.4 
 # После этого попробовал gensim + catboost, метрика ~0.55

        
class ModelConfig:
    NB_EPOCHS = 4
    LR = 3e-5
    EPS = 1e-8
    MAX_LEN = 32
    N_SPLITS = 4
    TRAIN_BS = 64
    VALID_BS = 32
    # Выбрал эту модель, так как обучалась для задачи мультиклассовой классификации,
    # но вообще не нашел сильно подходящей модели на huggingface
    MODEL_NAME = 'nbroad/ESG-BERT'
    TOKENIZER = BertTokenizer.from_pretrained('nbroad/ESG-BERT')

  
class LibDataset(Dataset):
    def __init__(self, texts, targets=None, le_classes=None, n_classes=24, is_test=False):
        self.texts = texts
        self.targets = targets
        self.le_classes = le_classes
        self.n_classes = n_classes
        self.is_test = is_test
        self.tokenizer = ModelConfig.TOKENIZER
        self.max_len = ModelConfig.MAX_LEN
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = ' '.join(text.split())
       
        inputs = self.tokenizer(
                            text,
                            add_special_tokens=True,
                            max_length=self.max_len,
                            padding="max_length" ,
                            truncation = True ,
                            pad_to_max_length=True, 
                            )
        
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
#         token_type = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
     
        le_classes = torch.tensor(self.le_classes[idx], dtype=torch.float32)
        if self.is_test:
            
            return {
                'ids': ids,
                'mask': mask,
                'le_classes': le_classes
#                 'token_type': token_type,
            }
        else:    
#             one_hot = F.one_hot(torch.tensor(self.targets[idx], dtype=torch.long), self.n_classes)
            targets = torch.tensor(self.targets[idx], dtype=torch.long)
            return {
                'ids': ids,
                'mask': mask,
                'le_classes': le_classes,
#                 'token_type': token_type,
                'targets': targets
            }


    
def LibDataloader(df, le_classes, batch_size, is_test=False, n_classes=24):
    dataset = LibDataset(texts=df["title"].values,
                         targets=df["label"].values,
                         le_classes=df[le_classes].values,
                         is_test=False,
                         n_classes = n_classes)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader


class LibClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(ModelConfig.MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(24, 768)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(768, 768)
        self.out = nn.Linear(1536, n_classes)
        
        
    def forward(self, input_ids, attention_mask, le_classes):
        
        _, output = self.bert(input_ids, attention_mask)
        output = self.drop(output)
        
        output_2 = self.fc1(le_classes)
        output_2 = self.drop1(output_2)
        output_2 = self.fc2(output_2)
        
        output = torch.cat((output, output_2), 1)
        output = self.out(output)
    
        return output
    
    
def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def precision_micro(outputs, targets):
    pred = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
    targets = targets.cpu().detach().numpy()
    return precision_score(pred, targets, average='micro')
    
  
def yield_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_parameters, lr=ModelConfig.LR, eps=ModelConfig.EPS) 
    
    
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = []

    for step , d in enumerate(data_loader):
        input_ids = d['ids'].to(device) 
        attention_mask = d['mask'].to(device)
        targets = d['targets'].to(device)
        le_classes = d['le_classes'].to(device)

        outputs = model(
            input_ids ,
            attention_mask ,
            le_classes
            )
        
        loss = loss_fn(outputs, targets)
        correct_predictions.append(precision_micro(outputs, targets))
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return np.mean(correct_predictions, axis=0), np.mean(losses)
    
    
def eval_model (model, data_loader, loss_fn, device, n_examples):
    model.eval()
  
    losses = []
    correct_predictions = []

    with torch.no_grad():
        for step , d in enumerate(data_loader):
            
            input_ids = d['ids'].to(device)
            attention_mask = d['mask'].to(device)
            targets = d['targets'].to(device)
            le_classes = d['le_classes'].to(device)
            outputs = model(
                    input_ids ,
                    attention_mask ,
                    le_classes
                )

            loss = loss_fn(outputs , targets)
            correct_predictions.append(precision_micro(outputs, targets))
            losses.append(loss.item())

    return np.mean(correct_predictions, axis=0), np.mean(losses)
    
    
def train(model, df, test_df, le_classes, epochs, device, model_name):
    best_accuracy = 0
    
    test_data_loader = LibDataloader(test_df, le_classes, ModelConfig.VALID_BS)
    
    for epoch in range(epochs):
        # if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}')

        kf = StratifiedKFold(n_splits=ModelConfig.N_SPLITS, random_state=seed, shuffle=True)

        for step, (train, valid ) in enumerate(kf.split(df , df["label"])) :

            train_data_loader = LibDataloader(df.iloc[train], le_classes, ModelConfig.TRAIN_BS)
            validation_data_loader = LibDataloader(df.iloc[valid], le_classes, ModelConfig.VALID_BS)

            nb_train_steps = int(len(train_data_loader) / ModelConfig.TRAIN_BS * epochs)
            optimizer = yield_optimizer(model)
            scheduler = get_linear_schedule_with_warmup(
                                        optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=nb_train_steps)

            train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df.iloc[train])) 
            val_acc, val_loss = eval_model(model, validation_data_loader, loss_fn, device, len(df.iloc[valid]))

            test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(test_df))
            
            if  test_acc > best_accuracy:
                torch.save(model.state_dict(), model_name + '.bin')
                best_accuracy = test_acc
                print(f"Best accuracy {best_accuracy}")
            
            
            

            
       
def get_predictions(model, df, le_classes):
    proba = []    
    model = model.eval()

    predictions = []

    test_data_loader = LibDataloader(df, le_classes, 1)
    with torch.no_grad():
        for d in test_data_loader:
            
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            le_classes = d['le_classes'].to(device)
            outputs = model(
                            input_ids,
                            attention_mask ,
                            le_classes
                            )
            proba.append(torch.argmax(outputs).flatten().cpu().numpy())
    return np.array(proba).flatten()
    

# def main():
     # seed all
seed = 42
seed_all(seed)

# import data
train_df = pd.read_csv('train.csv')
for_prediction_df = pd.read_csv('test.csv')

# encode labels and add features
le = LabelEncoder()
le.fit(train_df['lib'])
train_df['label'] = le.transform(train_df['lib'])
train_df = preprocess_data(train_df, le)

# Идея добавить классы пришла ко мне за пару часов до отпаравки,
# когда я посмотрел на предсказания модели на тестовом датасете.
# Но, кажется она не особо удачная (результаты вышли хуже)
# Поменять на предыдущую версию я не успеваю 
for_prediction_df = preprocess_data(for_prediction_df, le)
for_prediction_df['label'] = -1

# split for final model accuracy evaluation
train_data, test_data = train_test_split(train_df, train_size=0.9, stratify=train_df['label'])

train_data = train_data.sample(100)
test_data = train_data.sample(100)

# init model and start training
n_labels = len(train_df['label'].value_counts())
model = LibClassifier(n_labels)
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_data, test_data, le.classes_, ModelConfig.NB_EPOCHS, device, 'final_model')

# get submission
sample_submission = for_prediction_df[['id']]
sample_submission['lib'] = le.inverse_transform(get_predictions(model, for_prediction_df, le.classes_))
sample_submission.to_csv('./submission.csv')

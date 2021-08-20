---
title: "[boostcamp AI Tech] 학습기록 day13 (week3)"
date: 2021-08-19 22:50:05 -0400
categories:
use_math: true
---

# Pytorch
## Model load and save
1. Model.save
    * 학습 결과를 저장
    * 모델의 parameter와 buffer의 값들을 저장
    * 외부로 학습된 모델을 공유

    * checkpoint
        * 학습 중간 결과를 저장하여 early stopping등에 응용 및 학습 데이터 보존

2. Pretrained model and Transfer learning
    * 남이 저장한 모델을 사용하기 위한 방법
    * 이미 대용량의 데이터로 학습된 모델을 현재 application에 맞게 fine-tunning
    * backbone architecture에서 일부만 변경
    * freezing
        * pretrained model을 활용할 때 일부분의 parameter의 학습을 동결

## Monitoring tools
* Tensorboard
    * TensorFlow에서 만들어진 시각화 도구
    * 강력한 그래프 metric 학습결과 시각화로 pytorch에서도 연결하여 사용
    * scalar, graph, histogram, image, mesh

* weight & biases
    * 머신러닝 실험을 지원하기 위한 상용도구
    * 협업, code versioning, 실험결과 기록등의 기능 제공
    * MLOps의 대표적인 툴로 저변을 확대 중

# 과제
## 필수과제2 Custom Dataset 및 DataLoader 생성
* 커스텀 데이터셋 로더를 작성
* dataset loader의 기본적이 init, len, getitem을 각기 다른 3가지 타입의 데이터셋 로더를 작성하였다.
### TitanicDataset
```
class TitanicDataset(Dataset):
    def __init__(self, path, drop_features, train=True):
        self.data = pd.read_csv(path)
        self.data['Sex'] = self.data['Sex'].map({'male':0, 'female':1})
        self.data['Embarked'] = self.data['Embarked'].map({'S':0, 'C':1, 'Q':2})
        self.data = self.data.drop(drop_features, axis = 1)

        self.y = self.data['Survived'].values
        self.X = self.data.drop(['Survived'], axis = 1).values
        self.features = list(self.data.drop(['Survived'], axis = 1).columns)
        self.classes = ['Dead', 'Survived']

    def __len__(self):
        len_dataset=None
        len_dataset = len(self.y)
        return len_dataset

    def __getitem__(self, idx):
        X, y = None, None
        X = self.X[idx]
        y = self.y[idx]
        return torch.tensor(X), torch.tensor(y)
```
### Mnist
```
class MyMNISTDataset(Dataset):
    def __init__(self, path, transform, train=True):
        self.path = {'image':TRAIN_MNIST_IMAGE_PATH, 'label': TRAIN_MNIST_LABEL_PATH}

        self.X = read_MNIST_images(self.path['image'])
        self.y = read_MNIST_labels(self.path['label'])

        self.classes = list(set(self.y))
        self._repr_indent = 4
        self.transform = transform

    def __len__(self):
        len_dataset = None
        len_dataset = len(self.y)
        return len_dataset

    def __getitem__(self, idx):
        X,y = None, None
        X, y = self.transform(self.X[idx]), self.y[idx]
        return torch.tensor(X, dtype=torch.double), torch.tensor(y, dtype=torch.long)
```
### AG news
```
class MyAG_NEWSDataset(Dataset):
    def __init__(self, path='./data/AG_NEWS/train.csv', train=True):
        tqdm.pandas()
        self._repr_indent = 4
        self.data = pd.read_csv(path, sep=',', header=None, names=['class','title','description'])
        self.path = path
        self.train = train

        self.X = [title + " " + disc for title, disc in data.drop('class', axis=1).values]
        self.X = [self._preprocess(x) for x in self.X]
        self.y = data['class'].values
        self.classes = ['World', 'Sports', 'Business', 'Sci/Tech']

        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        self.counter = collections.Counter()
        for line in self.X:
            self.counter.update(self.tokenizer(line))
        self.vocab = torchtext.vocab.vocab(self.counter, min_freq=1)

        self.encoder = vocab.get_stoi()
        self.decoder = vocab.get_itos()

    def __len__(self):
        len_dataset = None
        len_dataset = len(self.y)
        return len_dataset

    def __getitem__(self, idx):
        X,y = None, None
        if self.train:
            X,y = self.X[idx], self.y[idx]
        else:
            return self.X[idx]
        return y, X 

    def _preprocess(self, s):   # 특수문자 제거
        words = s.split()
        words_after = []
        for word in words:
            word = "".join([c.lower() if c.isalnum() else " " for c in word ])
            words_after.append(word)
        s = " ".join(words_after)
        return s.replace("  ", " ").strip()
```
* 데이터 특성에 따라 중간에 가공해주는 과정이 비슷한 과정을 가졌다.
* 하지만 AG news의 경우 시퀀스 데이터라 그런지 앞의 두개와 다른 부분이 있어서 어려움을 겪었지만 앞에 예시 코드들을 참고하여 구현하였다.

# [피어세션 - 팀회고록](https://hackmd.io/@ai17/S1ppkFoxF)

# 후기
 CV를 해봤으므로 전체적으로 익숙한 과제였는데 AG news는 데이타셋의 특성이나 encoder와 decoder, vocab이라던가 생소한 개념들이 있어서 조금 시간이 걸렸다. 앞의 코드들을 보고도 vocab.get_stoi()와 vocab.get_itos()가 dict로 생성되는걸 늦게 알아차린게 컸다. mnist야 개념이 익숙하여 금방 하였지만 아무래도 과제 내에 NLP나 CV에 대한 개념, 특성 설명이 부족한데 NLP가 좀더 그런 부분이 부각된 느낌이었다.
'''
A simple Word2Vec implementation, for fun :)
'''

from collections import Counter
import random
import re
from switchblade.utils import infinite_dataloader
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, emb_dim, vocab2idx):
        super().__init__()
        self.vocab2idx = vocab2idx
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.context = nn.Embedding(vocab_size, emb_dim)
        self.similarity = lambda a, b: (a * b).sum(dim=-1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, words, contexts, labels):
        words = self.embeddings(words)
        contexts = self.context(contexts)
        similarity = self.similarity(words, contexts)
        loss = self.criterion(similarity, labels)
        return loss
    
    def get_embedding(self, word):
        idx = torch.tensor(self.vocab2idx[word], dtype=torch.long)
        return self.embeddings(idx)


def get_dataset(filename):
    # load data, remove punctuation, lowercase, and split
    with open(filename, 'r') as file:
        text = file.read()
    return re.sub(r'[^\w\s]', '', text.lower()).split()


def get_vocab(filename, n=None):
    words = get_dataset(filename)
    word_counts = Counter(words)
    vocab = word_counts.most_common(n)
    return [word for word, _ in vocab]


class SkipGramDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window = window_size

    def __getitem__(self, i):
        word = self.data[i]
        before = self.data[i-self.window:i]
        after = self.data[i+1:i+self.window+1]
        context = random.choice(before + after)

        return word, context

    def __len__(self):
        return len(self.data)


def plot_embeddings(embeddings, labels):
    df = pd.DataFrame(embeddings, columns=['x', 'y'])
    df['label'] = labels
    fig = px.scatter(df, x='x', y='y', text='label')
    fig.show()


if __name__ == '__main__':
    # constants and hyperparameters
    filename = 'jp.txt'
    vocab_size = 10000
    emb_dim = 128
    batch_size = 1024
    lr = 3e-4
    n_iters = 50000
    window_size = 10
    negative_ratio = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create dataset
    vocab = get_vocab(filename, vocab_size)
    vocab_size = len(vocab)
    vocab2idx = {word : i for i, word in enumerate(vocab)}
    data = [vocab2idx[w] for w in get_dataset(filename) if w in vocab2idx]
    dataset = SkipGramDataset(data, window_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    dataloader = infinite_dataloader(dataloader)

    # create model and optimizer
    model = Word2Vec(vocab_size, emb_dim, vocab2idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train the model
    losses = []
    progress_bar = trange(n_iters)
    model.train()
    for _ in progress_bar:
        words, contexts = next(dataloader)
        words = words.to(device)
        contexts = contexts.to(device)
        n_negatives = len(words) * negative_ratio
        negatives = torch.randint(0, vocab_size, (n_negatives,)).to(device)

        positive_label = torch.ones_like(words).float()
        negative_label = torch.zeros_like(negatives).float()
        words_for_negative = words.repeat(negative_ratio)[:n_negatives]

        loss = model(words, contexts, positive_label)
        loss += model(words_for_negative, negatives, negative_label)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_description(f'Loss: {loss.item():.3f}')

    print('Model training complete!')

    # visualize data
    if emb_dim != 2:
        embeds = TSNE().fit(
            model.embeddings.weight.detach().numpy()
        ).embedding_
    else:
        embeds = model.embeddings.weight.detach().numpy()

    plot_embeddings(embeds, vocab)
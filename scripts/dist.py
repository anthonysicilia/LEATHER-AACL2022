import torch
import json
import random
from math import sqrt
from sklearn.cluster import KMeans

# get energy distance

class ModEnergyStatistic:

    def __init__(self, n_1, n_2, nclusters=100, seed=0):

        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = -1. / (n_1 * n_1)
        self.a11 = -1. / (n_2 * n_2)
        self.a01 = 1. / (n_1 * n_2)

        self.nclusters = nclusters
        self.seed = seed

    def __call__(self, sample_1, sample_2, train_sample_1, train_sample_2):

        train_sample_12 = torch.cat((train_sample_1, train_sample_2), 0)
        coarsening_fn = KMeans(n_clusters=self.nclusters,
            random_state=self.seed).fit(train_sample_12)

        sample_12 = torch.cat((sample_1, sample_2), 0)
        cats = torch.tensor(coarsening_fn.predict(sample_12))
        cats = cats.expand(cats.size(0), cats.size(0))
        cdist = (cats != cats.transpose(0, 1)).float()
        # dist = pdist(sample_12, sample_12, norm=2)
        for i in range(cdist.size(0)): cdist[i, i] = 0
        # for i in range(dist.size(0)): dist[i, i] = 0
        # dist = (dist > 0).float()
        d_1 = cdist[:self.n_1, :self.n_1].sum()
        d_2 = cdist[-self.n_2:, -self.n_2:].sum()
        d_12 = cdist[:self.n_1, -self.n_2:].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        return loss

def get_vecs(x, shuffle=False):
    x = json.load(open(x, 'rb'))
    ws = []
    for k in x.keys():
        ws.append(x[k]['enc_hidden'])
    if shuffle:
        random.shuffle(ws)
    return torch.tensor(ws).squeeze(1)

if __name__ == '__main__':

    refs = [
        'logs/GamePlay/f0s2e652022_05_11_11_44/test_GPinference_f0s2e65_2022_05_11_11_44.json',
        'logs/GamePlay/f0s2e752022_05_11_11_28/test_GPinference_f0s2e75_2022_05_11_11_28.json',
        'logs/GamePlay/f0s2e852022_05_11_11_11/test_GPinference_f0s2e85_2022_05_11_11_11.json',
        'logs/GamePlay/f0s2e952022_04_29_09_04/test_GPinference_f0s2e95_2022_04_29_09_04.json',

        'logs/GamePlay/f1s2e652022_05_11_11_46/test_GPinference_f1s2e65_2022_05_11_11_46.json',
        'logs/GamePlay/f1s2e752022_05_11_11_29/test_GPinference_f1s2e75_2022_05_11_11_29.json',
        'logs/GamePlay/f1s2e852022_05_11_11_11/test_GPinference_f1s2e85_2022_05_11_11_11.json',
        'logs/GamePlay/f1s2e952022_04_29_09_03/test_GPinference_f1s2e95_2022_04_29_09_03.json'
    ]

    comps = [
        'logs/GamePlay/f0s2e662022_05_11_11_51/test_GPinference_f0s2e66_2022_05_11_11_51.json',
        'logs/GamePlay/f0s2e762022_05_11_11_36/test_GPinference_f0s2e76_2022_05_11_11_36.json',
        'logs/GamePlay/f0s2e862022_05_11_11_20/test_GPinference_f0s2e86_2022_05_11_11_20.json',
        'logs/GamePlay/f0s2e962022_04_28_10_49/test_GPinference_f0s2e96_2022_04_28_10_49.json',

        'logs/GamePlay/f1s2e662022_05_11_11_54/test_GPinference_f1s2e66_2022_05_11_11_54.json',
        'logs/GamePlay/f1s2e762022_05_11_11_37/test_GPinference_f1s2e76_2022_05_11_11_37.json',
        'logs/GamePlay/f1s2e862022_04_28_10_49/test_GPinference_f1s2e86_2022_04_28_10_49.json',
        'logs/GamePlay/f1s2e962022_04_29_09_12/test_GPinference_f1s2e96_2022_04_29_09_12.json',
    ]

    N = 5_000

    for clusters in [10, 50, 100, 500, 1000]:
        for ref, comp in zip(refs, comps):
            stats = []
            for seed in [1, 2, 3]:
                random.seed(seed)
                a = get_vecs(ref, shuffle=True)
                b = get_vecs(comp, shuffle=True)
                stat = ModEnergyStatistic(N, N, nclusters=clusters, seed=seed)
                stats.append(sqrt(stat(a[:N], b[:N], a[N:2*N], b[N:2*N]).item()))
            print(comp.split('2022')[0], clusters, 'energy', sum(stats) / len(stats))
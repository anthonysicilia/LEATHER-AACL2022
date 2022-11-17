import pandas as pd

# process human evals

if __name__ == '__main__':
    df = pd.read_csv('annot-clean-all.txt')
    data = dict()

    for idx, shortcut, rel, spec, num in zip(df['id'], df['shortcut'], df['rel'], df['spec'], df['num']):
        if idx not in data:
            data[idx] = dict()
        if shortcut not in data[idx]:
            data[idx][shortcut] = dict()
        data[idx][shortcut]['rel'] = rel
        data[idx][shortcut]['spec'] = spec
        data[idx][shortcut]['num'] = num
    
    diffs = {
        'f1s2e86' : {
            'rel' : [],
            'spec' : []
        },
        'f0s2e96' : {
            'rel' : [],
            'spec' : []
        }
    }

    vals = {
        'f1s2e86' : {
            'rel' : [],
            'spec' : []
        },
        'f0s2e96' : {
            'rel' : [],
            'spec' : []
        }
    }

    for i in data.keys():
        for shortcut in data[i].keys():
            if shortcut != 'human':
                for k in ['rel', 'spec']:
                    v = data[i][shortcut][k] / 8
                    x = abs(data[i][shortcut][k] / 8 - data[i]['human'][k] / data[i]['human']['num'])
                    diffs[shortcut][k].append(x)
                    vals[shortcut][k].append(v)

    print('diffs')
    for k1 in diffs.keys():
        for k2 in diffs[k1].keys():
            print(k1, k2, sum(diffs[k1][k2]) / len(diffs[k1][k2]))
    
    print('vals')
    for k1 in diffs.keys():
        for k2 in diffs[k1].keys():
            print(k1, k2, sum(vals[k1][k2]) / len(vals[k1][k2]))
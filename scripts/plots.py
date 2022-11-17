import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'font.size' : 18})

# builds plots for paper

xticks = [65, 75, 85, 95]

no_human = {
    '10' : [0.3073789642, 0.2272832378, 0.3170764829, 0.1587992379],
    '50' : [0.1869599696, 0.1903099559, 0.1943452027, 0.1794606033],
    '100' : [0.1427313953, 0.1536381635, 0.1436394069, 0.1329172344],
    '500' : [0.08179429121, 0.083951492, 0.08259088071, 0.07941708029],
    '1000' : [0.06649416368, 0.06748332832, 0.06719510134, 0.0659696184],
    'lexical div.' : [0.01, 0.008, 0.003, 0.004],
    'rep. question' : [0.065, 0.058, 0.061, -0.01],
    'ques. types' : [0.00501, 0.00709, 0.0211, -0.00963]
}

fifty_fifty = {
    '10' : [0.1066787034, 0.04497283575, 0.02963784567, 0.09680972624],
    '50' : [0.1480561381, 0.1213273828, 0.1354816849, 0.1850796201],
    '100' : [0.1222228736, 0.1199743515, 0.1099193178, 0.1389408127],
    '500' : [0.0777525719, 0.07351699402, 0.06777124611, 0.07334806093],
    '1000' : [0.06408164534, 0.06425414945, 0.05864423063, 0.06198831438],
    'lexical div.' : [-0.034, -0.032, -0.019, -0.023],
    'rep. question' : [-0.105, -0.093, -0.096, -0.088],
    'ques. types' : [-0.01298, -0.01099, -0.0247, -0.01269]
}

if __name__ == '__main__':

    fig, ax = plt.subplots(3, 5, figsize=(17, 9))
    i = 0
    for g in ['lexical div.', 'rep. question', 'ques. types']:
        for k in ['10', '50', '100', '500', '1000']:
            ax.flat[i].scatter(no_human[k], no_human[g], label='CL', s=80, marker='o')
            ax.flat[i].scatter(fifty_fifty[k], fifty_fifty[g], label='Ours', s=80, marker='^')
            x = np.array(no_human[k] + fifty_fifty[k])
            y = np.array(no_human[g] + fifty_fifty[g])
            a, b = np.polyfit(x, y, 1)
            ax.flat[i].plot(x, a*x+b, ls=':', lw=4)
            ax.flat[i].set_title(f'{g}')
            if i % 5 == 0:
                ax.flat[i].set_ylabel('change')
            # if i == 0:
                # ax.flat[i].set_ylim((-0.2, 0.33))
                # ax.flat[i].set_xlim((0.10, 0.24))
                # ax.flat[i].legend(loc='lower right')
            if i >= 10:
                ax.flat[i].set_xlabel(f'energy (k={k})')
            i += 1
    plt.tight_layout()
    plt.savefig('energy-full'); plt.clf()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    i = 0
    k = '100'
    for g in ['lexical div.', 'rep. question', 'ques. types']:
        ax.flat[i].scatter(no_human[k], no_human[g], label='CL', s=80, marker='o')
        ax.flat[i].scatter(fifty_fifty[k], fifty_fifty[g], label='Ours', s=80, marker='^')
        x = np.array(no_human[k] + fifty_fifty[k])
        y = np.array(no_human[g] + fifty_fifty[g])
        a, b = np.polyfit(x, y, 1)
        ax.flat[i].plot(x, a*x+b, ls=':', lw=4)
        ax.flat[i].set_title(f'test={g}')
        if i == 0:
            ax.flat[i].set_ylabel('change in ber')
        # if i == 0:
            # ax.flat[i].legend(loc='lower right')
        ax.flat[i].set_xlabel(f'energy')
        i += 1
    plt.tight_layout()
    plt.savefig('energy'); plt.clf()

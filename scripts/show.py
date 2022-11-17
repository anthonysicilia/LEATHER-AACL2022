import json
import random
import gzip

# test, show data to human annotator to asnwer questions

def print_dial(s):
    print(s
        .replace('<start>', '')
        .replace('>', '>\n') 
        .replace('<yes>', 'Yes')
        .replace('<no>', 'No')
        .replace(' ?', '?'),
        end='')

if __name__ == '__main__':
    file_name = 'data/guesswhat.test.jsonl.gz'
    file = gzip.open(file_name)
    hdata = dict()
    for line in file:
        game = json.loads(line.decode("utf-8"))
        hdata[game['id']] = game

    # with gzip.open(file_name) as file:
    #     for line in file:
    #         game = json.loads(line.decode("utf-8"))
    #         print(f'============')
    #         print(f'Dialogue (human)')
    #         x = ''
    #         for turns in game['qas']:
    #             x += turns['question']
    #             x += ' ' + turns['answer'] + '\n'
    #         print_dial(x)
    #         print(f'============')
    #         break

    files = [
        'logs/GamePlay/f1s2e862022_04_28_10_49/test_GPinference_f1s2e86_2022_04_28_10_49.json',
        'logs/GamePlay/f0s2e962022_04_29_09_12/test_GPinference_f0s2e96_2022_04_29_09_12.json',

        # 'logs/GamePlay/f1s2e652022_05_11_11_46/test_GPinference_f1s2e65_2022_05_11_11_46.json',
        # 'logs/GamePlay/f1s2e662022_05_11_11_54/test_GPinference_f1s2e66_2022_05_11_11_54.json',
        # 'logs/GamePlay/f1s2e752022_05_11_11_29/test_GPinference_f1s2e75_2022_05_11_11_29.json',
        # 'logs/GamePlay/f1s2e762022_05_11_11_37/test_GPinference_f1s2e76_2022_05_11_11_37.json',
        # 'logs/GamePlay/f1s2e852022_05_11_11_11/test_GPinference_f1s2e85_2022_05_11_11_11.json',
        # 'logs/GamePlay/f1s2e862022_05_11_11_20/test_GPinference_f1s2e86_2022_05_11_11_20.json',
        # 'logs/GamePlay/f1s2e952022_04_29_09_03/test_GPinference_f1s2e95_2022_04_29_09_03.json',
        # 'logs/GamePlay/f1s2e962022_04_29_09_12/test_GPinference_f1s2e96_2022_04_29_09_12.json',

        # 'logs/GamePlay/f0s2e652022_05_11_11_44/test_GPinference_f0s2e65_2022_05_11_11_44.json',
        # 'logs/GamePlay/f0s2e662022_05_11_11_51/test_GPinference_f0s2e66_2022_05_11_11_51.json',
        # 'logs/GamePlay/f0s2e752022_05_11_11_28/test_GPinference_f0s2e75_2022_05_11_11_28.json',
        # 'logs/GamePlay/f0s2e762022_05_11_11_36/test_GPinference_f0s2e76_2022_05_11_11_36.json',
        # 'logs/GamePlay/f0s2e852022_05_11_11_11/test_GPinference_f0s2e85_2022_05_11_11_11.json',
        # 'logs/GamePlay/f0s2e862022_05_11_11_20/test_GPinference_f0s2e86_2022_05_11_11_20.json',
        # 'logs/GamePlay/f1s2e952022_04_29_09_03/test_GPinference_f1s2e95_2022_04_29_09_03.json',
        # 'logs/GamePlay/f1s2e962022_04_29_09_12/test_GPinference_f1s2e96_2022_04_29_09_12.json'
    ]

    # names = ['proposed', 'baseline']
    names = ['B', 'C']
    data = json.load(open(files[0], 'r'))
    keys = list(data.keys())
    random.shuffle(keys)
    key = keys[0]

    match = data[key]
    imid = int(match['image'].split('_')[-1].split('.')[0])
    caid = match['target_cat_id']
    with gzip.open(file_name) as file:
        for line in file:
            game = json.loads(line.decode("utf-8"))
            output = game['image']['id'] == imid
            if not output:
                continue
            goid = game['object_id']
            gcaid = None
            for obj in game['objects']:
                if goid == obj['id']:
                    gcaid = obj['category_id']
                    print(key)
                    print('Image:', imid)
                    print('Goal:', obj['category'])
            output = output and (gcaid == caid)
            if output:
                print(f'============')
                print(f'Dialogue A')
                # print(f'Dialogue (human)')
                x = ''
                for turns in game['qas']:
                    x += turns['question']
                    x += ' ' + turns['answer'] + '\n'
                print_dial(x)
                print(f'============')
                break

    for i, file in enumerate(files):
        # print(names[i])
        if i > 0:
            data = json.load(open(file, 'r'))
        x = data[key]
        print(f'============')
        print(f'Dialogue {names[i]}')
        print_dial(x['gen_dialogue'])
        print(f'============')
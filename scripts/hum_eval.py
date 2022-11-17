import json
import random
import gzip

# official, show data to human annotator

keys = [
    '85499', '112212', '104010', '154147', '107137', 
    '107788', '83352', '61577', '215804', '158138', 
    '56897', '74724', '164658', '133965', '34329', 
    '129479', '113176', '18354', '178033', '117953', 
    '109821', '83005', '216796', '149793', '182188', 
    '183513', '36139', '120484', '217671', '81755', 
    '93216', '157580', '8549', '79865', '181020', 
    '29992', '166922', '147344', '161797', '87653', 
    '180641', '210145', '41034', '3813', '196678', 
    '187551', '177861', '52555', '101981', '126382', 
    '44658', '162834', '100057', '187512', '167243', 
    '82599', '173718', '39777', '204725', '219826', 
    '157365', '166352', '147411', '78097', '11868', 
    '123933', '74566', '113099', '188801', '221625', 
    '208985', '204994', '174505', '159995', '203833', 
    '169387', '22667', '37924', '181290', '52199', 
    '123036', '162109', '200888', '68870', '38568', 
    '97217', '97280', '21432', '205580', '86369', 
    '165230', '143798', '174165', '147322', '97246', 
    '222157', '35276', '129121', '218136', '144606'
]

files = {
    'f1s2e86' : 'logs/GamePlay/f1s2e862022_04_28_10_49/test_GPinference_f1s2e86_2022_04_28_10_49.json',
    'f0s2e96' : 'logs/GamePlay/f0s2e962022_04_29_09_12/test_GPinference_f0s2e96_2022_04_29_09_12.json',

    # 'f1s2e65' : 'logs/GamePlay/f1s2e652022_05_11_11_46/test_GPinference_f1s2e65_2022_05_11_11_46.json',
    # 'f1s2e66' : 'logs/GamePlay/f1s2e662022_05_11_11_54/test_GPinference_f1s2e66_2022_05_11_11_54.json',
    # 'f1s2e75' : 'logs/GamePlay/f1s2e752022_05_11_11_29/test_GPinference_f1s2e75_2022_05_11_11_29.json',
    # 'f1s2e76' : 'logs/GamePlay/f1s2e762022_05_11_11_37/test_GPinference_f1s2e76_2022_05_11_11_37.json',
    # 'f1s2e85' : 'logs/GamePlay/f1s2e852022_05_11_11_11/test_GPinference_f1s2e85_2022_05_11_11_11.json',
    # 'f1s2e86' : 'logs/GamePlay/f1s2e862022_05_11_11_20/test_GPinference_f1s2e86_2022_05_11_11_20.json',
    # 'f1s2e95' : 'logs/GamePlay/f1s2e952022_04_29_09_03/test_GPinference_f1s2e95_2022_04_29_09_03.json',
    # 'f1s2e96' : 'logs/GamePlay/f1s2e962022_04_29_09_12/test_GPinference_f1s2e96_2022_04_29_09_12.json',

    # 'f0s2e65' : 'logs/GamePlay/f0s2e652022_05_11_11_44/test_GPinference_f0s2e65_2022_05_11_11_44.json',
    # 'f0s2e66' : 'logs/GamePlay/f0s2e662022_05_11_11_51/test_GPinference_f0s2e66_2022_05_11_11_51.json',
    # 'f0s2e75' : 'logs/GamePlay/f0s2e752022_05_11_11_28/test_GPinference_f0s2e75_2022_05_11_11_28.json',
    # 'f0s2e76' : 'logs/GamePlay/f0s2e762022_05_11_11_36/test_GPinference_f0s2e76_2022_05_11_11_36.json',
    # 'f0s2e85' : 'logs/GamePlay/f0s2e852022_05_11_11_11/test_GPinference_f0s2e85_2022_05_11_11_11.json',
    # 'f0s2e86' : 'logs/GamePlay/f0s2e862022_05_11_11_20/test_GPinference_f0s2e86_2022_05_11_11_20.json',
    # 'f0s2e95' : 'logs/GamePlay/f0s2e952022_04_29_09_04/test_GPinference_f0s2e95_2022_04_29_09_04.json',
    # 'f0s2e96' : 'logs/GamePlay/f0s2e962022_04_29_09_12/test_GPinference_f0s2e96_2022_04_29_09_12.json',

    'human' : 'data/guesswhat.test.jsonl.gz',
}

def print_dial(s):
    print(s
        .replace('<start>', '')
        .replace('>', '>\n') 
        .replace('<yes>', 'Yes')
        .replace('<no>', 'No')
        .replace(' ?', '?'),
        end='')

if __name__ == '__main__':

    aid = input('What is your annotator ID? (#): ')
    aid = int(aid)
    start = input('What round would you like to start at? (1-100): ')
    start = int(start)
    print('Preparing data, this make take a while...')
    annotator_happy = True
    data = dict()

    random.seed(0)

    for i, key in enumerate(keys):

        if i < start - 1:
            continue

        dials = dict()

        for j, (shortcut, file) in enumerate(files.items()):

            if shortcut != 'human':
                if shortcut not in data:
                    data[shortcut] = json.load(open(file, 'r'))
                if j == 0:
                    imid = int(data[shortcut][key]['image'].split('_')[-1].split('.')[0])
                    caid = data[shortcut][key]['target_cat_id']
                dials[shortcut] = data[shortcut][key]['gen_dialogue']

        # special processing for human
        file = files['human']
        shortcut = 'human'
        file = gzip.open(file)
        if shortcut not in data:
            temp = dict()
            for line in file:
                game = json.loads(line.decode("utf-8"))
                temp[game['id']] = game
            data[shortcut] = temp
        for game in data[shortcut].values():
            output = game['image']['id'] == imid
            if not output:
                continue
            goid = game['object_id']
            gcaid = None
            for obj in game['objects']:
                if goid == obj['id']:
                    gcaid = obj['category_id']
                    premise_str = f"Round {i+1}\nImage {imid}\nGoal {obj['category']}"
            output = output and (gcaid == caid)
            if output:
                x = ''
                for turns in game['qas']:
                    x += turns['question']
                    x += ' ' + turns['answer'] + '\n'
                dials['human'] = x
                break
        
        dials = list(dials.items())
        random.shuffle(dials)
        for k, v in dials:
            print(premise_str)
            print_dial(v)
            relevance = input('Relevance? #Q w/ Errors: ')
            relevance = int(relevance)
            spec = input('Specificty? #Q w/ Mult. Descriptors: ')
            spec = int(spec)
            gram = -1
            top = -1
            # gram = input('Grammar? #Q w/ Errors: ')
            # gram = int(gram)
            # top = input('Topic Maintenance? #Q w/ Errors: ')
            # top = int(top)
            with open(f'annotator-{aid}.txt', 'a') as out:
                n = len(v.split('\n')) # num human questions
                out.write(f'{k}, {v}, {relevance}, {spec}, {gram}, {top}, {n}\n')
            with open(f'annotator-{aid}-clean.txt', 'a') as out:
                n = len(v.split('\n')) # num human questions
                out.write(f'{k}, {relevance}, {spec}, {gram}, {top}, {n}\n')
        annotator_happy = input('Continue? (Y/N)')
        annotator_happy = (annotator_happy.upper() == 'Y')
        if not annotator_happy:
            print('Last round:', i+1, '...exiting...')
            exit()
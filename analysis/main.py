import json
from nltk.tokenize import TweetTokenizer
import copy
import collections

from qclassify import qclass
from math import log, exp
import gzip
import random

def split_que_ans(game, human=False):
    new_game = copy.deepcopy(game)
    for key in game:
        new_game[key] = copy.deepcopy(game[key])
        if human:
            gen_dialogue = [game[key]['true_dialogue']]
            gen_dialogue[0] = gen_dialogue[0].replace('Yes','<Yes>')
            gen_dialogue[0] = gen_dialogue[0].replace('yes','<yes>')
            gen_dialogue[0] = gen_dialogue[0].replace('No','<No>')
            gen_dialogue[0] = gen_dialogue[0].replace('no','<no>')
            gen_dialogue[0] = gen_dialogue[0].replace('NA','<NA>')
            gen_dialogue[0] = gen_dialogue[0].replace('na','<na>')
            new_game[key]['gen_dialogue'] = copy.deepcopy(game[key]['true_dialogue'])
        else:
            gen_dialogue = [game[key]['gen_dialogue']] 
        gen_dialogue[0] = gen_dialogue[0].replace('<start>','')
        
        # replace < and > in unk token so we can split on that one next
        gen_dialogue[0] = gen_dialogue[0].replace('<unk>', '_unk_')
        
        tmp_gen_dialogue = [dial.split('>') for dial in gen_dialogue]
        
        new_game[key]['que'] = []
        new_game[key]['ans'] = []
        for dialogue in tmp_gen_dialogue[0]:
            dialogue = dialogue.replace('<','')
            if dialogue:
                try:
                    que,ans = dialogue.split('?')
                    que = que + ' ?'
                    new_game[key]['que'].append(que)
                    new_game[key]['ans'].append(ans.lower().strip())
                except:
                    pass
                    # nothing = 1
    # print('Spliting is done')                
    return(new_game)

def divstats_single_game(ques):

    tknzr = TweetTokenizer(preserve_case=False)
    ttr = 0
    q_tokens = list()

    game_que = []
    game_rep_flag = False

    for que in ques:

        q_tokens.extend(tknzr.tokenize(que))
        if que in game_que:
            game_rep_flag = True
            
        game_que.append(que)

    ttr = len(set(q_tokens)) / len(q_tokens)

    return ttr, int(game_rep_flag)

def one_fn_to_rule_them_all(hum_game, gen_game):
    classifier = qclass()

    humstats = dict()
    for hgame in hum_game:
        cats = []
        ques = []
        for _, qa in enumerate(hgame['qas']):
            que = qa['question']
            ques.append(que)
            que = que.lower()
            cat = classifier.que_classify_multi(que)
            if type(cat) == list:
                cats.extend(cat)
            elif type(cat) == str:
                cats.append(cat)
        ldiv, rq = divstats_single_game(ques)
        img = hgame['image']['file_name']
        if img not in humstats:
            humstats[img] = []
        humstats[img].append({'cats' : cats, 'ldiv' : ldiv, 'rq' : rq})
    
    genstats = dict()
    for key in gen_game:
        cats = []
        ques = []
        for _, que in enumerate(gen_game[key]['que']):
            ques.append(que)
            que = que.lower()
            cat = classifier.que_classify_multi(que)
            if type(cat) == list:
                cats.extend(cat)
            elif type(cat) == str:
                cats.append(cat)
        ldiv, rq = divstats_single_game(ques)
        img = gen_game[key]['image']
        if img not in genstats:
            genstats[img] = []
        genstats[img].append({'cats' : cats, 'ldiv' : ldiv, 'rq' : rq})
    
    stats = {'ldiv' : [], 'rq' : []}
    allcats = set([cat 
        for img in humstats 
        for x in humstats[img] 
        for cat in x['cats']])
        
    for cat in allcats:
        stats[cat] = []
    
    for img in humstats:
        if img in genstats:
            harr = humstats[img]
            garr = genstats[img]
            random.shuffle(harr)
            random.shuffle(garr)
            for h, g in zip(harr, garr):
                for k in ['ldiv', 'rq']:
                    stats[k].append(abs(h[k] - g[k]))
                for cat in allcats:
                    p = sum([cat == c for c in h['cats']]) / len(h['cats'])
                    q = sum([cat == c for c in g['cats']]) / len(g['cats'])
                    stats[cat].append(abs(p - q))
    
    # print(len(stats['ldiv']))

    for k, v in stats.items():
        stats[k] = sum(v) / len(v)
    
    return stats
                
def get_que_type_multi(game): #, q_max_miss=0, file_name='dummy.json'):
    
    # Classification of game
    
    # ATTRIBUTES:
        # - color
        # - shape
        # - size
        # - texture
        # - action
        # - spatial/location
       
    # ENTITY
        # - super-category
        # - object

    # tknzr = TweetTokenizer(preserve_case=False)
    classifier = qclass()
    q_count = 0
    
    d_count =0
    # number_count = 0
    color_count = 0
    shape_count = 0
    size_count = 0
    texture_count = 0
    action_count = 0
    spatial_count = 0
    object_count = 0
    super_count = 0
    
    att_count = 0
    
    new_game = copy.deepcopy(game)
    
    for key in game:
        new_game[key]['q_ann'] = [] #To store categoty
        new_game[key]['q_ann_at'] = [] #To store broad categoty
        
        d_count +=1
        # d_flag = True
        # d_flag_count = 0
        questions = game[key]['que']
        game[key]['gen_dialogue'] = game[key]['gen_dialogue'].replace('?', ' ?')
        # dial =  [n for x, n in enumerate(game[key]['gen_dialogue'].split())]
    
        for _, que in enumerate(questions):
            q_count +=1
            cat = '<NA>'
            que = que.lower()
            cat = classifier.que_classify_multi(que)
            
            att_flag = False
            
            if '<color>' in cat:
                color_count +=1
                att_flag = True
                
            if '<shape>' in cat:
                shape_count +=1
                att_flag = True
            if '<size>' in cat:
                size_count +=1
                att_flag = True
            if '<texture>' in cat:
                texture_count +=1
                att_flag = True
            if '<action>' in cat:
                action_count +=1
                att_flag = True
            if '<spatial>' in cat:
                spatial_count +=1
                att_flag = True
            if att_flag:
                att_count +=1
                
            if '<object>' in cat:
                object_count +=1
            if '<super-category>' in cat:
                super_count +=1
                
            new_game[key]['q_ann'].append(cat)
            if cat == '<NA>' or cat == '<object>' or cat == '<super-category>':
                new_game[key]['q_ann_at'].append(cat)
            else:
                new_game[key]['q_ann_at'].append("<attribute>")
#             break
            
    
    ent_count = object_count + super_count
    
    ent_per = ent_count * 100 / q_count
    object_per = object_count * 100 / q_count
    super_per = super_count * 100 / q_count
    
    att_per = att_count * 100 / q_count
    color_per = color_count * 100 / q_count
    shape_per = shape_count * 100 / q_count
    size_per = size_count * 100 / q_count
    texture_per = texture_count * 100 / q_count
    spatial_per = spatial_count * 100 / q_count
    action_per = action_count * 100 / q_count
    rest_per = 100 - (att_per+ent_per) 
    
#     print('Entity : ', ent_per)
#     print('Super : ', super_per)
#     print('Object : ', object_per)
#     print('Attribute : ', att_per)
#     print('Color : ', color_per)
#     print('Shape : ', shape_per)
#     print('Size : ', size_per)
#     print('Texture : ', texture_per)
#     print('Location : ', spatial_per)
#     print('Action : ', action_per)
#     print('NA : ', rest_per)
    
    out = {}
    out['Entity']= ent_per
    out['Super ']=  super_per
    out['Object']= object_per
    out['Attribute']= att_per
    out['Color']= color_per
    out['Shape']= shape_per
    out['Size']= size_per
    out['Texture']= texture_per
    out['Location']= spatial_per
    out['Action']= action_per
    out['NA']= rest_per
    # print(out)
    return out, new_game

def lexicalDiversity(game, maxQ=-1):
    # LexicalDiversity as type token ratio https://www.sltinfo.com/type-token-ratio/
    # maxQ in case we want to analysis on part of the dialogue i.e. 5Q, 6Q only
    tknzr = TweetTokenizer(preserve_case=False)
    ttr = 0
    q_tokens = list()
    all_que = []
    for key in game:
        ques = game[key]['que']
        anss = game[key]['ans']

        q_count = 0

        for que, ans in zip(ques, anss):

            if maxQ > 0 and q_count >=maxQ:
                break
            q_tokens.extend(tknzr.tokenize(que))

            q_count +=1
    ttr = len(set(q_tokens)) * 100 / len(q_tokens)

    # avg_ttr = ttr / len(game)
    # print(ttr, len(q_tokens),len(set(q_tokens)), len(all_que))
    return ttr

def questionDiversity(game, maxQ=-1, human=False):
    # Question Diversity and % of Game with repeated questions
    # maxQ in case we want to analysis on part of the dialogue i.e. 5Q, 6Q only
    all_que = []

    game_rep = 0
    
    #Default word in the Vocabulary
    vocab = ['<padding>',
              '<start>',
              '<stop>',
              '<stop_dialogue>',
              '<unk>',
              '<yes>' ,
              '<no>',
              '<n/a>',
            ]

    if human:
        word2i = {'<padding>': 0,
                  '<start>': 1,
                  '<stop>': 2,
                  '<stop_dialogue>': 3,
                  '<unk>': 4,
                  '<yes>': 5,
                  '<no>': 6,
                  '<n/a>': 7,
                  }


        min_occ = 1
        word2occ = collections.OrderedDict()
        tknzr = TweetTokenizer(preserve_case=False)

        for key in game:
            questions = game[key]['que']
            q_count = 0
            for que_idx, que in enumerate(questions):
                if maxQ > 0 and q_count >= maxQ:
                    continue

                tokens = tknzr.tokenize(que)
                for tok in tokens:
                    if tok not in word2occ:
                        word2occ[tok] = 1
                    else:
                        word2occ[tok] += 1
        for word, occ in word2occ.items():
            if occ >= min_occ and word.count('.') <= 1:
                word2i[word] = len(word2i)
        # print(len(word2i))

    all_q_count = 0
    for key in game:
        questions = game[key]['que']

        game_rep_flag = False
        game_que = []

        q_count = 0
        for que_idx, que in enumerate(questions):

            if maxQ > 0 and q_count >=maxQ:
                continue

            words = que.split()


            for word in words:
                if word not in vocab:
                    vocab.append(word)

            if que not in all_que:
                all_que.append(que)
            if que in game_que:
                game_rep_flag = True
            
            game_que.append(que)
            
            q_count += 1
            all_q_count +=1
        if game_rep_flag:
            game_rep += 1
        

    # print(len(vocab))
    num_unique_que = len(all_que)
    per_rep_game = game_rep * 100 / len(game)
    len_vocab = len(vocab)

    out = {}
    out['num_que'] = all_q_count
    out['num_unique_que'] = num_unique_que
    out['que_divesity'] = num_unique_que*100/all_q_count
    out['%_rep_game'] = per_rep_game
    out['len_vocab'] = len_vocab

    # print('all_que_count =',all_q_count )
    return out

def normalize_keys(x, keys):
    m = max([v for k,v in x.items() if k in keys])
    s = sum([exp(v - m) for k,v in x.items() if k in keys])
    for k in keys:
        x[k] = exp(x[k] - m) / s
    return x

def accuracy(game):
    correct = 0
    for key in game:
        if game[key]['target_id'] == game[key]['guess_id']:
            correct += 1
    return correct / len(game)

def round3sig(x):
    return f'{float(f"{x:.3g}"):g}'

if __name__ == '__main__':

    paths = [

        # substitute your log files here
        # 'logs/GamePlay/f0s2e652022_05_11_11_44/test_GPinference_f0s2e65_2022_05_11_11_44.json',
        # 'logs/GamePlay/f0s2e752022_05_11_11_28/test_GPinference_f0s2e75_2022_05_11_11_28.json',
        # 'logs/GamePlay/f0s2e852022_05_11_11_11/test_GPinference_f0s2e85_2022_05_11_11_11.json',
        # 'logs/GamePlay/f0s2e952022_04_29_09_04/test_GPinference_f0s2e95_2022_04_29_09_04.json',

        # 'logs/GamePlay/f1s2e652022_05_11_11_46/test_GPinference_f1s2e65_2022_05_11_11_46.json',
        # 'logs/GamePlay/f1s2e752022_05_11_11_29/test_GPinference_f1s2e75_2022_05_11_11_29.json',
        # 'logs/GamePlay/f1s2e852022_05_11_11_11/test_GPinference_f1s2e85_2022_05_11_11_11.json',
        # 'logs/GamePlay/f1s2e952022_04_29_09_03/test_GPinference_f1s2e95_2022_04_29_09_03.json',

        # 'logs/GamePlay/f0s2e662022_05_11_11_51/test_GPinference_f0s2e66_2022_05_11_11_51.json',
        # 'logs/GamePlay/f0s2e762022_05_11_11_36/test_GPinference_f0s2e76_2022_05_11_11_36.json',
        # 'logs/GamePlay/f0s2e862022_05_11_11_20/test_GPinference_f0s2e86_2022_05_11_11_20.json',
        # 'logs/GamePlay/f0s2e962022_04_28_10_49/test_GPinference_f0s2e96_2022_04_28_10_49.json',

        # 'logs/GamePlay/f1s2e662022_05_11_11_54/test_GPinference_f1s2e66_2022_05_11_11_54.json',
        # 'logs/GamePlay/f1s2e762022_05_11_11_37/test_GPinference_f1s2e76_2022_05_11_11_37.json',
        # 'logs/GamePlay/f1s2e862022_04_28_10_49/test_GPinference_f1s2e86_2022_04_28_10_49.json',
        # 'logs/GamePlay/f1s2e962022_04_29_09_12/test_GPinference_f1s2e96_2022_04_29_09_12.json'

    ]

    for file_name in paths:
        # file_name = '../../data/dummy.json'
        print(file_name.split('2022')[0].split('/')[-1])
        with open(file_name) as file:
            game = json.load(file)

        print(round3sig(accuracy(game)))

        game = split_que_ans(game)

        # Table Results
        lexDiv = lexicalDiversity(game)
        queDiv = questionDiversity(game)
        print(round3sig(lexDiv))
        print(round3sig(queDiv['que_divesity']))
        print(round3sig(queDiv['%_rep_game']))

        # # Analyses for plots
        hum_games = []
        with gzip.open('data/guesswhat.test.jsonl.gz') as file:
            for json_game in file:
                hum_games.append(json.loads(json_game.decode("utf-8")))
        proxy = one_fn_to_rule_them_all(hum_games, game)
        proxy = {k : v for k, v in sorted(proxy.items(), key=lambda a: a[0])}
        for _, v in sorted(proxy.items(), key=lambda a: a[0]):
            print(round3sig(v))
        for _, d in proxy.items():
            print(round3sig(d['diff']))
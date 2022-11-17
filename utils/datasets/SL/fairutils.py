import json
import random

from string import punctuation

MALE = ['he', 'man', 'him', 'his', 'guy', 'boy', 'men', 'guys', 'boys']
FEMALE = ['she', 'woman', 'her', 'hers', 'gal', 'girl', 'women', 'gals', 'girls']

def ismale(s):
    for word in s.split():
        w = word.lower().strip(punctuation)
        if w in MALE:
            return True
    return False

def isfemale(s):
    for word in s.split():
        w = word.lower().strip(punctuation)
        if w in FEMALE:
            return True
    return False

def process_annotations():
    paths = ['annotations/captions_train2014.json', 
        'annotations/captions_val2014.json']
    captions = dict()
    for file_path in paths:
        with open(file_path, 'r') as file:
            annots = json.load(file)['annotations']
            for a in annots:
                if a['image_id'] not in captions:
                    captions[a['image_id']] = []
                captions[a['image_id']].append(a['caption'])
    images_w_male = []
    images_w_female = []
    for image, caps in captions.items():
        male = False
        female = False
        for c in caps:
            male = male or ismale(c)
            female = female or isfemale(c)
        if male: images_w_male.append(image)
        if female: images_w_female.append(image)
    return images_w_male, images_w_female

def downsample(a, b):
    while len(a) > len(b):
        i = random.choice(list(range(len(a))))
        del a[i]
    while len(b) > len(a):
        i = random.choice(list(range(len(b))))
        del b[i]
    return a, b
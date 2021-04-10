import json
import random
train = json.load(open("dataset.json", "r"))
items = [(v["audio_filepath"].split("/")[-2], (k, v)) for k, v in train.items()]
pairs = {}
for key, item in items:
    try:
        pairs[key].append(item)
    except:
        pairs[key] = [item]
train_pairs, test_pairs ={}, {}
for key in pairs.keys():
    random.shuffle(pairs[key])
    train_pairs.update(pairs[key][:100])
    test_pairs.update(pairs[key][100:])
json.dump(train_pairs, open("train.json", "w"), indent=4, ensure_ascii=False)
json.dump(test_pairs, open("test.json", "w"), indent=4, ensure_ascii=False)

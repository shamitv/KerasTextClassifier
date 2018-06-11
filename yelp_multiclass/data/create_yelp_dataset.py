import numpy as np
import re
import json
from yelp_multiclass.data.config import getDataFile,getJsonFile,getSentenceDir
from yelp_multiclass.data.SentencesClasses import getStarRating,getOneHotClass
from collections import OrderedDict

npz_file=getDataFile()
json_file=getJsonFile()
yelp_dir=getSentenceDir()

def repeat_array(arr,n):
    ret=[]
    for x in range(0,n):
        ret.append(arr)
    return np.asarray(ret,dtype=np.object_)

def read_sents_from_file(yelp_dir,prefix,sent_class):
    ret = {};
    suffixes = np.array(['test','training','validation']);
    for s in suffixes:
        filename=yelp_dir+prefix+s+".txt";
        print("Reading :: "+filename)
        with open(filename,"r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            #if(len(lines)>100):
            #    del lines[100:]
            ret[s]=lines;

    if 'training' in ret.keys():
        if 'validation' in ret.keys():
            validation_sents=ret['validation']
            del ret['validation']
            train_sents=ret['training']
            train_sents.extend(validation_sents)

    review_class_str=getStarRating(sent_class)
    one_hot_class=getOneHotClass(review_class_str)
    training_y=repeat_array(one_hot_class,len(ret['training']))
    test_y = repeat_array(one_hot_class, len(ret['test']))
    ret['training_y']=training_y
    ret['test_y'] = test_y
    return ret;


sents_1_star=read_sents_from_file(yelp_dir,"1_star_",1)
sents_2_star=read_sents_from_file(yelp_dir,"2_star_",2)
sents_3_star=read_sents_from_file(yelp_dir,"3_star_",3)
sents_4_star=read_sents_from_file(yelp_dir,"4_star_",4)
sents_5_star=read_sents_from_file(yelp_dir,"5_star_",5)

print("Loaded 1 to 5 star sentences");

all_sents=[]

data=[sents_1_star,sents_2_star,sents_3_star,sents_4_star,sents_5_star]

for d in data:
    all_sents.extend(d['test']);
    all_sents.extend(d['training']);

print(str(len(all_sents))+" sentences ")

word_counts={}

for s in all_sents:
    tokens=re.split("[\\s\\.!\\?]+",s);
    for t in tokens:
        ts=t.lower();
        if(ts in word_counts.keys()):
            count=word_counts[ts];
            count += 1;
            word_counts[ts]=count;
        else:
            word_counts[ts] = 1;

print("Total words = "+str(len(word_counts.keys())))

sorted_touples=sorted(word_counts.items(),  key=lambda x: x[1],reverse=True);

sorted_word_count =  OrderedDict(sorted_touples);

word_idx=OrderedDict();
index=1;
for word in sorted_word_count.keys():
    word_idx[word]=index;
    index+=1;

print("Total words in sorted = "+str(len(sorted_word_count.keys())))

def replace_words_with_indices(sent_list):
    ret=[]
    for s in sent_list:
        tokens = re.split("[\\s\\.!\\?]+", s);
        vals = []
        for t in tokens:
            ts = t.lower();
            idx = word_idx[ts]
            vals.append(idx)
        ret.append(vals)
    return ret;


x_train=[]
x_test=[]
y_train=[]
y_test=[]

for d in data:
    x_train.extend(replace_words_with_indices(d['training']))
    y_train.extend(d['training_y'])
    x_test.extend(replace_words_with_indices(d['test']))
    y_test.extend(d['test_y'])

x_train=np.asarray(x_train,dtype=np.object_)
x_test=np.asarray(x_test,dtype=np.object_)
y_train=np.asarray(y_train,dtype=np.object_)
y_test=np.asarray(y_test,dtype=np.object_)

print(y_train[20])

npz_dict={'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test}

np.savez(npz_file,**npz_dict)

print("Saved file :: "+npz_file)

with open(json_file, 'w', encoding="utf-8") as fp:
    json.dump(word_idx,fp, indent=4)

print("Saved file :: " + json_file)
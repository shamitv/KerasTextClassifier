import re

def tokenizeText(text):
    x=text.lower();
    tokens = re.split("[\\s\\.!\\?]+", x);
    return tokens;


def buildIndexToWordDict(word_idx):
    ret={}
    for w in word_idx.keys():
        ret[word_idx[w]]=w
    return ret;


def createTextSent(x,idx_word):
    ret="";
    for val in x:
        ret = ret + " " + idx_word[val];
    return ret;


def createNumericSent(x,word_idx):
    ret=[]
    tokens=tokenizeText(x)
    for t in tokens:
        if(t in word_idx.keys()):
            ret.append(word_idx[t])
        else:
            # by convention, use 2 as OOV word
            # reserve 'index_from' (=3 by default) characters:
            # 0 (padding), 1 (start), 2 (OOV)
            ret.append(2)
    return ret;
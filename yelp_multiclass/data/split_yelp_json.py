import numpy as np
import re
import json
import random
from yelp_multiclass.data.config import getDataFile,getJsonFile,getSentenceDir,getYelpJsonFile
from yelp_multiclass.data.SentencesClasses import getStarRating,getOneHotClass
from collections import OrderedDict

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

log = logging.getLogger('Yelp_Review_Processor')
log.setLevel(logging.DEBUG)

json_path=getYelpJsonFile()

log.info("Processing "+json_path)

reviews={"1":[],"2":[],"3":[],"4":[],"5":[]}

with open(json_path,"r", encoding="utf-8") as f:
    for line in f:
        j=json.loads(line)
        review_list=reviews[str(j["stars"])]
        review_list.append(j["text"])


log.info("Parsed "+json_path)
print(len(reviews["1"]))
print(len(reviews["2"]))
print(len(reviews["3"]))
print(len(reviews["4"]))
print(len(reviews["5"]))

negative_reviews= reviews["1"]+reviews["2"]
positive_reviews= reviews["4"]+reviews["5"]

min_reviews=min(len(negative_reviews),len(positive_reviews))

log.info("Count of samples = "+str(min_reviews*2))

random.shuffle(negative_reviews)
random.shuffle(positive_reviews)

log.info("samples shuffled")

train_count=0.6*min_reviews
test_count=val_count=0.2*min_reviews






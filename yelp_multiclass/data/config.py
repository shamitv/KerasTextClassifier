base_dir="G:\\work\\nlp\\datasets\\yelp\\yelp_dataset_challenge_round9\\keras\\"
yelp_dir=base_dir+"sentences\\"
dataset_dir=base_dir+"dataset\\"
npz_file=dataset_dir+"yelp_review_mullticlass.npz";
json_file=dataset_dir+"yelp_word_index_mullticlass.json"
model_file=dataset_dir+"yelp_model_mullticlass.h5"


def getDataFile():
    return npz_file


def getJsonFile():
    return json_file;


def getModelFile():
    return model_file;

def getSentenceDir():
    return yelp_dir;
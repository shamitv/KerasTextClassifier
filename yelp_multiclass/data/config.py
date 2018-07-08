base_dir="F:\\nlp\\yelp\\"
yelp_dir=base_dir+"sentences\\"
dataset_dir=base_dir+"dataset\\"
model_dir=base_dir+"model\\"
log_dir=base_dir+"log\\"
npz_file=dataset_dir+"yelp_review_mullticlass.npz";
json_file=dataset_dir+"yelp_word_index_mullticlass.json"
model_file=model_dir+"yelp_model_mullticlass.h5"



def getDataFile():
    return npz_file


def getJsonFile():
    return json_file;


def getModelFile():
    return model_file;

def getSentenceDir():
    return yelp_dir;

def getModelDir():
    return model_dir;

def getLogDir():
    return log_dir;
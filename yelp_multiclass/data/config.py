base_dir="F:\\nlp\\yelp\\"
yelp_dir=base_dir+"sentences2\\"
dataset_dir=base_dir+"dataset\\"
model_dir=base_dir+"model\\"
log_dir=base_dir+"log\\"
npz_file=dataset_dir+"yelp_review_binary.npz";
processed_npz_file=dataset_dir+"yelp_review_binary_processed.npz";
json_file=dataset_dir+"yelp_word_index_binary.json"
model_file=model_dir+"yelp_model_mullticlass.h5"
yelp_source_json=dataset_dir+"source\\review_round11.json"


def getDataFile():
    return npz_file

def getProcessedDataFile():
    return processed_npz_file

def getJsonFile():
    return json_file;

def getYelpJsonFile():
    return yelp_source_json;

def getModelFile():
    return model_file;

def getSentenceDir():
    return yelp_dir;

def getModelDir():
    return model_dir;

def getLogDir():
    return log_dir;
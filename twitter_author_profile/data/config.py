base_dir="G:\\work\\nlp\\datasets\\pan_twitter\\pan16\\"
english_dir=base_dir+"english\\"
dataset_dir=base_dir+"dataset\\"
npz_file=dataset_dir+"pan16_twitter.npz";
json_file=dataset_dir+"pan16_twitter.json"
model_file=dataset_dir+"pan16_twitter.h5"


def getDataFile():
    return npz_file


def getJsonFile():
    return json_file;


def getModelFile():
    return model_file;

def getSentenceDir():
    return yelp_dir;
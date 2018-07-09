import numpy as np

from yelp_multiclass.data.yelp_dataset import load_data, load_word_indices
from yelp_multiclass.data.textProcess import tokenizeText,buildIndexToWordDict,createTextSent,createNumericSent
from yelp_multiclass.data.config import getDataFile


NUM_WORDS = 300000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 256
NUM_EPOCHS = 3  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5

(X_train, y_train), (X_test, y_test) = load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

sample= X_train[0]

print(sample)

word_index = load_word_indices()
index_word = buildIndexToWordDict(word_index)
words = list(map(index_word.get,sample))

print(words)

sample = [x-1 for x in sample]
sample[0]=1
print(sample)
words = list(map(index_word.get,sample))
print(words)


negative_review=" Its a roadside restaurant (pathetic)...its no way a cafe! If u end up visiting...you'll be served disgust! Its a part of some theatre and there's some alien name on the board"
positive_review="The food was absolutely wonderful, from preparation to presentation, very pleasing.. especially the veg cheese salad and falafal. Truly a home like taste away from home!!. thumbs up. "

num_x=createNumericSent(negative_review,word_index)
print(num_x)

idx_word=buildIndexToWordDict(word_index)
print(createTextSent(num_x,idx_word))

path=getDataFile()

with np.load(path) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']

sample = x_train[200]
print(sample)
words = list(map(index_word.get,sample))

print(words)

import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

labels=["1","2","3","4","5","1","2","3","4","5","1","2","3","4","5","1","2","3","4","5","1","2","3","4","5",
        "1","2","3","4","5","1","2","3","4","5","1","2","3","4","5"]

encoder = LabelEncoder()
encoder.fit(labels)

y = encoder.transform(labels).astype(np.int32)
y = np_utils.to_categorical(y)

print(y)


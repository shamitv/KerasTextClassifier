import numpy as np
from collections import OrderedDict

classOneHotMap=OrderedDict([
        ("1-star" , np.array([1, 0, 0, 0, 0],dtype=np.int32)),
        ("2-star" , np.array([0, 1, 0, 0, 0],dtype=np.int32)),
        ("3-star" , np.array([0, 0, 1, 0, 0],dtype=np.int32)),
        ("4-star" , np.array([0, 0, 0, 1, 0],dtype=np.int32)),
        ("5-star" , np.array([0, 0, 0, 0, 1],dtype=np.int32))
    ])

keys=classOneHotMap.keys()
class_keys=[]
for x in keys:
    class_keys.append(x)


argmax = lambda iterable, func: max(iterable, key=func)
def getOneHotClass(class_str):
    if class_str in classOneHotMap.keys():
        return classOneHotMap[class_str]
    else:
        raise ValueError('Unknown text class :: '+class_str)

def getStringClass(one_hot):
    if(len(one_hot)==5):
        key_idx =  np.argmax(one_hot)
        return class_keys[key_idx]
    else:
        raise ValueError('Invalid One Hot array, size should be 5 :: ' + one_hot)

def getStarRating(star_count):
    return str(star_count)+"-star"
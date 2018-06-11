from yelp_multiclass.data.SentencesClasses import getOneHotClass,getStringClass,getStarRating

for x in range(1, 6):
    str_class=getStarRating(x)
    one_hot=getOneHotClass(str_class)
    print(str_class +" => " + str(one_hot)+ " => "+getStringClass(one_hot))


#!/usr/bin/python
"""
Example of attention coefficients visualization

Uses saved model, so it should be executed after train.py
"""
import os
from yelp_sentiment_attention.train import *
from yelp_multiclass.data.yelp_dataset import load_word_indices

from yelp_multiclass.data.textProcess import createTextSent,tokenizeText,buildIndexToWordDict

saver = tf.train.Saver()

# Calculate alpha coefficients for the first test example
with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)

    x_batch_test, y_batch_test = X_test[:550], y_test[:550]
    seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
    results = sess.run([alphas,y_hat], feed_dict={batch_ph: x_batch_test, target_ph: y_batch_test,
                                                seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
alphas_values = results[0][0]


word_index = load_word_indices()
index_word = buildIndexToWordDict(word_index)




#textSent = createTextSent(x_batch_test[0]);
#words=tokenizeText(textSent);

# Save visualization as HTML
with open("visualization.html", "w") as html_file:
    for X,alpha_values,Y_hat,Y in zip(x_batch_test,results[0],results[1],y_batch_test ) :
        html_file.write("<br/><hr/>");
        sample = [x - 1 for x in X]
        sample[0] = 1
        sample=[0 if x == -1 else x for x in sample]
        words = list(map(index_word.get, sample))
        html_file.write('%s  %s\n' % (Y, Y_hat));
        if Y_hat > 0:
            div_color='#b8efc8'
        else:
            div_color = '#ff848a'
        html_file.write('<div style="background-color:%s">\n' % div_color);
        for word, alpha in zip(words, alphas_values / alphas_values.max()):
            if word == ":START:":
                continue
            elif word == ":PAD:":
                break
            html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
        html_file.write('</div>\n');
print('Working director is ' +  os.getcwd())

print('\nOpen visualization.html to checkout the attention coefficients visualization.')

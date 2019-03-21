import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random 
import pickle
from collections import Counter 

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            content = f.readlines()
            for l in content[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon) # -> w_counts would be something like a count for word dictionary {'the':40000,'and':20000}
    l2 = []
    for w in w_counts:
        if 1000  > w_counts[w] > 50: # we dont want the super common words, like nothing special meanings
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample,lexicon,classification): # classification here is more like a label
    featuresets = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_word = word_tokenize(l.lower())
            current_word = [lemmatizer.lemmatize(i) for i in current_word]
            features = np.zeros(len(lexicon))
            for word in current_word:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featuresets.append([features,classification])
    return featuresets

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size]) # -> [:,0] : [[5,8],[7,9]] -> [:,0] returns [5,7]
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

# the final question : does tf.argmax([output]) == tf.argmax([expectations])
# tf.argmax([53974,34382]) == tf.argmax([0,1])
# if not shuffle , [9999999999999,-99999999999999] would be the output thing, shifting the weights to make the argmax to be true

if __name__ == "__main__":
    train_x, train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)


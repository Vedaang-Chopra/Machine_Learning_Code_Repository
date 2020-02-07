from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import movie_reviews
import random
from nltk.corpus import wordnet
import nltk


def loading():
    # print(movie_reviews.categories())
    # print(movie_reviews.fileids())
    documents=[]
    for i in movie_reviews.categories():
        for j in movie_reviews.fileids():
            documents.append((movie_reviews.words(j),i))
    # print(documents[0:5])
    random.shuffle(documents)
    return documents


def pos_to_wordnet(pos_tag):
    # print(pos_tag)
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('R'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN


def cleaning_words(words):
    word_array=[]
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    punctuations = list(string.punctuation)
    stop_words += punctuations
    for i in range(len(words)):
        if words[i] in stop_words:
            continue
        else:
            pos_tuple_returned=pos_tag([words[i]])
            word=lemmatizer.lemmatize(words[i],pos=pos_to_wordnet(pos_tuple_returned[0][1]))
            word_array.append(word)
    return word_array


def cleaning_file(document):
    cleaned_words=[]
    for i in range(len(document)):
        new_array=cleaning_words(document[i][0])
        print(new_array)
        cleaned_words.append((new_array,document[i][1]))
    return cleaned_words

def finding_features(document):
    feature_words=[]
    for i in range(len(document)):
        feature_words.append(document[i][0])
    freq= nltk.FreqDist(feature_words)
    top_words_tuple=freq.most_common(3000)
    top_words=[i[0] for i in top_words_tuple]
    print(top_words)
    return top_words


def creating_dictionary_single_file(top_words,words):
    dict={}
    for i in range(len(words)):
        if words[i] in top_words:
            dict+={ words[i] : True }
        else:
            continue
    return dict

def creating_dictionary(top_words,document):
    feature_set=[]
    for i in range(len(document)):
        dict=creating_dictionary_single_file(top_words,document[i][0])
        feature_set.append((dict,document[i][1]))

    return feature_set

from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier


document=loading()
print(document)
cleaned_words=cleaning_file(document)
x_y_train, x_y_test=cleaned_words[0:1500],cleaned_words[1500:]
top_words=finding_features(x_y_train)
feature_set=creating_dictionary(top_words,x_y_train)
test_set=creating_dictionary(top_words,x_y_test)
from nltk import NaiveBayesClassifier
classifier=NaiveBayesClassifier.train(feature_set)
print(nltk.classify.accuracy(classifier,test_set))
print(classifier.show_most_informative_features(15))


svm=SVC()
classifier_sklearn=SklearnClassifier(svm)
random_forest=RandomForestClassifier()
classifier_sklearn_1=SklearnClassifier(random_forest)


classifier_sklearn.train(feature_set)
nltk.classify.accuracy(classifier_sklearn,test_set)


classifier_sklearn_1.train(feature_set)
nltk.classify.accuracy(classifier_sklearn_1,test_set)



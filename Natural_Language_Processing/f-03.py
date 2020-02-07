from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet
import random
from sklearn.feature_extraction.text import CountVectorizer


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

def creating_sentences(x_y_train):
    x_train_sentences=[]
    y_categories=[]
    for i in range(len(x_y_train)):
        x_train_sentences.append(" ".join(x_y_train[i][0]))
        y_categories.append(x_y_train[i][1])
    return x_train_sentences,y_categories

def count_vectorize(x_train_sentences,x_test_sentences):
    count_vec=CountVectorizer(max_features=2000)
    x_train_features=count_vec.fit_transform(x_train_sentences)
    print(count_vec.get_feature_names())
    print(x_train_features.todense())
    x_test_features=count_vec.transform(x_test_sentences)
    return x_train_features,x_test_features

document=loading()
print(document)
cleaned_words=cleaning_file(document)
x_y_train, x_y_test=cleaned_words[0:1500],cleaned_words[1500:]
x_train_sentences,y_train_categories=creating_sentences(x_y_train)
x_test_sentences,y_test_categories=creating_sentences(x_y_test)
x_train_features,x_test_features=count_vectorize(x_train_sentences,x_test_sentences)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_features,y_train_categories)
y_pred=rf.predict(x_test_features)
from  sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test_categories,y_pred))
print(confusion_matrix(y_test_categories,y_pred))


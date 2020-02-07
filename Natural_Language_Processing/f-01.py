# import nltk
# nltk.download ()
# Tokenising..............................................
from nltk.tokenize import sent_tokenize,word_tokenize
sample_text="Joey says, 'How you doing!!'"
sample_text="Does this really work? Lets see."
sent_arr=sent_tokenize(sample_text)
word_arr=word_tokenize(sample_text.lower())

# Stopwords and Punctuations.........................................
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
print(stop_words)
import string
punctuations=list(string.punctuation)
print(string.punctuation)
stop_words+=punctuations
# Cleaning words(removing stopwords and punctuations
clean_words=[w for w in word_arr if not w in stop_words]

# Stemming...........................................
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stem_words=['play','playing','played','player',"happy",'happier']
stemmed_words=[ps.stem(w) for w in stem_words]
print(stem_words)

# Parts of a speech......................................
from nltk.corpus import state_union
from nltk import pos_tag
speech_george_bush_2006=state_union.raw('2006-GWBush.txt')
parts_of_speech=pos_tag(word_tokenize(speech_george_bush_2006))
print(parts_of_speech)

# Lemmatization..........................................
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
a1=lemmatizer.lemmatize('good',pos='n')
a2=lemmatizer.lemmatize('good',pos='a')
a3=lemmatizer.lemmatize('better',pos='a')
a4=lemmatizer.lemmatize('excellent',pos='a')
a5=lemmatizer.lemmatize('paint',pos='n')
a6=lemmatizer.lemmatize('painting',pos='n')
a7=lemmatizer.lemmatize('painting',pos='v')
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a6)
print(a7)
from nltk.corpus import wordnet
def pos_to_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('M'):
        return wordnet.MODAL
    elif pos_tag.startswith('R'):
        return wordnet.ADVERB
    else:
        return wordnet.NOUN



import src.data.tokenizer as tokenizer

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
import re

def NonNerCleanText(data):

    # basic tokenizer 
    B_tokenizer = tokenizer.PlainTokenizer()

    # set stop words
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
    # punctuations = string.punctuation.replace('-','')
    lemmatizer = WordNetLemmatizer()

    data = re.sub(r'\([^()]*\)', '', data)
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')
    words = [s for s in B_tokenizer.tokenize(data) if re.match("^[A-Za-z0-9\-]+$", s)]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(s) for s in words]
    words = [s for s in words if not re.match("^[0-9]+$", s)]
    words = [s for s in words if not len(s) == 1]

    # Only keeping the nouns 
    is_noun = lambda pos: pos[:2] == 'NN'
    words = [word for (word, pos) in pos_tag(words) if is_noun(pos) or '-' in word] 

    return words

def NERCleanText(key, data, path2entity):
    """
        This method cleans the text and creates tokens 
        @param filepath: path to articles 
        @oaram path2entity: ner dictionary 

        @return words: nested list; 
                       each inner list contains the tokens of each article 
    """
    # using ner tokenizer which keep ner terms intact 
    B_tokenizer = tokenizer.NERTokenizer()

    # set stop words 
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
    punctuations = string.punctuation
    lemmatizer = WordNetLemmatizer() 

    # load NER entities
    entities = list(path2entity[key])
    entities.sort(key=len, reverse=True)    

    # open file read data 
    

    # non-words tag
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')

    # check NER terms
    terms = []
    for t in entities:

        # replace " " with "_" for NER terms
        if ' ' in t:
            tnew = t.replace(' ', '_')
            data = data.replace(t, tnew)
            terms.append(tnew)
        else:
            terms.append(t)
    terms.sort(key=len, reverse=True)

    # using NERTokenizer
    words = [s for s in B_tokenizer.tokenize(data, terms) if not (len(s) == 1 and (s in punctuations))]
    
    # remove stop words 
    words = [w for w in words if not w in stop_words]
    words_new = []
    for w in words:

        # if ner terms, keep intact 
        if w in terms:
            words_new.append(w.replace('_', ' '))

        # else lemmatize
        else:
            words_new.append(lemmatizer.lemmatize(w))

    words = words_new
    words = [s for s in words if not re.match("^[0-9]+$", s)]
    words = [s for s in words if not len(s) == 1]
    return words
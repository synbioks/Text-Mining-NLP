import preprocess as preprocess
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from nltk.stem import WordNetLemmatizer
import re

def NonNerCleanText(data):

    # basic tokenizer 
    B_tokenizer = preprocess.PlainTokenizer()

    # set stop words
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
    # punctuations = string.punctuation.replace('-','')
    lemmatizer = WordNetLemmatizer()

    data = re.sub(r'\([^()]*\)', '', data)
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')
    words = [s for s in B_tokenizer.tokenize(data) if re.match("^[A-Za-z0-9]+$", s)]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(s) for s in words]
    words = [s for s in words if not re.match("^[0-9]+$", s)]
    words = [s for s in words if not len(s) == 1]

    # Only keeping the nouns 
    is_noun = lambda pos: pos[:2] == 'NN'
    words = [word for (word, pos) in pos_tag(words) if is_noun(pos)] 

    return words

def NERCleanText(filepath, path2entity, separate, lemmatization):
    """
        This method cleans the text and creates tokens 
    """
    # using ner tokenizer which keep ner terms intact 
    B_tokenizer = preprocess.NERTokenizer()

    # set stop words 
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union({'et', 'al', 'use', 'using', 'used'})
    punctuations = string.punctuation
    lemmatizer = WordNetLemmatizer() 
    entities = list(path2entity[filepath])
    entities.sort(key=len, reverse=True)
    with open(filepath, 'r') as f:
        data = f.read()
    for tag in ['REFEND', 'REF', 'EQL', 'FIG']:
        data = data.replace(tag, '')
    terms = []
    for t in entities:
        if ' ' in t:
            tnew = t.replace(' ', '_')
            data = data.replace(t, tnew)
            terms.append(tnew)
        else:
            terms.append(t)
    terms.sort(key=len, reverse=True)
    words = [s for s in B_tokenizer.tokenize(data, terms) if not (len(s) == 1 and (s in punctuations))]
    words = [w for w in words if not w in stop_words]
    words_new = []
    for w in words:
        if w in terms:
            if separate:
                words_new.extend(w.split('_'))
            else:
                words_new.append(w)
        else:
            if lemmatization:
                words_new.append(lemmatizer.lemmatize(w))
            else:
                words_new.append(w)
    words = words_new
    words = [s for s in words if not re.match("^[0-9]+$", s)]
    words = [s for s in words if not len(s) == 1]
    return words
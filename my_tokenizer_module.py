# my_tokenizer_module.py

import nltk
import string
import re
from nltk.corpus import stopwords

# Initialize stemmer and download NLTK resources
stemmer = nltk.stem.PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
ENGLISH_STOP_WORDS = stopwords.words('english')

def remove_html_tags(text):
    pattern = re.compile(r'<.*?>')  
    return pattern.sub('', text)

def my_tokenizer(sentence):
    # Remove punctuation and set to lowercase
    sentence = sentence.lower()
    sentence = ''.join([char for char in sentence if char not in string.punctuation])

    # Remove digits using list comprehension
    sentence = ''.join([char for char in sentence if not char.isdigit()])

    # Remove HTML tags
    sentence = remove_html_tags(sentence)

    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)

    # Remove stopwords and stem words
    list_of_stemmed_words = []
    for word in words:
        if (not word in ENGLISH_STOP_WORDS) and (word != ''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            list_of_stemmed_words.append(stemmed_word)

    return list_of_stemmed_words

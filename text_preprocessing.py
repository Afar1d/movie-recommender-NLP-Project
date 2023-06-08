import os
from nltk.corpus import stopwords, words


def clean_text(f):
    with open(f, 'r', encoding='utf-16') as f1:
        lines = f1.read()
        all_words = lines.split()  # split all words, but it not words only its contain number and other things

        # step 1 : to make words contain alphabetic only
        alphabetic_only = [word for word in all_words if word.isalpha()]  # get  alphabetic only

        # step 2 : remove non-english word
        words_nltk = set(words.words())  ## english word
        english_alphabetic_only = [word for word in alphabetic_only if word in words_nltk]

        # step 3 : convert all words into lowercase
        lower_case_only = [word.lower() for word in english_alphabetic_only]

        # step 4 : drop all the stop words
        stopwords_nltk = set(stopwords.words('english'))  ## all stop words
        clean_words = [word for word in lower_case_only if word not in stopwords_nltk]
        print(len(all_words))
        print(len(clean_words))
        f1.close()
    return clean_words

# ======================================== #
path = 'Skill_BOK (1).txt'
print(clean_text(path))


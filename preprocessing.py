import spacy
nlp = spacy.load('en_core_web_md')

from hunspell import Hunspell
h = Hunspell()

import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


from word import Word

# Remove stop words
def stop_word_stripper(line):
    stop_words = [w.strip('\n').lower() for w in open('stop_words.txt').readlines()]
    pos_stopper = ['PUNCT', 'SYM']
    return ' '.join([token.text for token in line if str(token).lower() not in stop_words and token.pos_  not in pos_stopper])


def assign_font_size(propercase_freq, max_size, min_size):
    label_fs = {}
    sorted_tuples = [(k, propercase_freq[k]) for k in sorted(propercase_freq, key=propercase_freq.get, reverse=True)]
    min_count = sorted_tuples[-1][1]
    max_count = sorted_tuples[0][1]

    for kw, count in sorted_tuples:
        if (max_count - min_count) == 0:
            size = int((max_size - min_size) / 2.0 + min_size)
        else:
            #size = int(min_size + (max_size - min_size) * (count * 1.0 / (max_count - min_count)) ** 0.8)
            size = int((max_size - min_size)/(max_count - min_count)*count + min_size - (max_size - min_size)/(max_count - min_count)*min_count)
        label_fs[kw] = size

    # return sorted label_fs in decreasing order of font_size
    return label_fs


def max_dimensions(kw_fs):
    kw_dimensions = {}
    for kw, fs in kw_fs.items():
        kw_dimensions[kw] = (int(0.65*len(kw)*fs), fs) #x, y (i.e. width, height)
    return kw_dimensions


def pre_processing(filename):

    docs = [x.strip('\n') for x in open('data/{}'.format(filename)).readlines()]

    stripped_docs = [] #spacy objects
    copy_docs = [] # strings

    for d in docs:
        stripped_docs.append(nlp(stop_word_stripper(nlp(d))))
        copy_docs.append(stop_word_stripper(nlp(d)))

    # parse through to get entities
    kw_freq = {}

    for i in range(len(stripped_docs)):
        line = stripped_docs[i]
        for e in line.ents:
            copy_docs[i] = copy_docs[i].replace(e.text, '').strip()
            if e.text in kw_freq:
                kw_freq[e.text] += 1
            else:
                kw_freq[e.text] = 1

    # get lemma keywords 
    # join the rest of the words together: 
    # spell check for only lower case words
    corpus = nlp(' '.join(copy_docs))

    MIN_CHARACTERS = 3

    for token in corpus:
        if len(token.lemma_) >= MIN_CHARACTERS:
            word = token.lemma_
            if word.lower == word: #only spell check lower cased words
                if not h.spell(token.lemma_):
                    if len(h.suggest(token.lemma_)) > 0:
                        word = h.suggest(token.lemma_)[0]
            if word in kw_freq:
                kw_freq[word] += 1
            else:
                kw_freq[word] = 1

    # proper casing
    caseless_freq = {}
    propercase_freq = {}

    for kw, count in kw_freq.items():
        if kw in caseless_freq:
            caseless_freq[kw.lower()].append(count)
        else:
            caseless_freq[kw.lower()] = [count]

    for kw, count in kw_freq.items():
        if count == max(caseless_freq[kw.lower()]):
            propercase_freq[kw] = sum(caseless_freq[kw.lower()])
    
    glove_vectors = []
    labels_array = []

    number_of_not_recignized_word = 0 
    for kw, count in propercase_freq.items():
        labels_array.append(kw)
        if nlp(kw)[0].vector.any() :
            glove_vectors.append(nlp(kw)[0].vector)
        else: #word not found
            number_of_not_recignized_word += 1
            glove_vectors.append(np.array([0]*300))

    # if the number over half the words are brand names finish with only one cluster 
    if number_of_not_recignized_word > len(glove_vectors)*0.5:
        kw_cluster = {}
        for label in labels_array:
            kw_cluster[label] = 0
    else:
        # AffinityPropagation clustering 
        AffinityPropagation_model = AffinityPropagation()
        AffinityPropagation_model.fit(glove_vectors)

        cluster_labels    = AffinityPropagation_model.labels_

        kw_cluster = {}
        for i in range(len(labels_array)):
            kw_cluster[labels_array[i]] = cluster_labels[i]


    #distance matrix (len(cluster_labels)^2)
    n = len(labels_array)
    distance_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = spatial.distance.cosine(glove_vectors[i], glove_vectors[j])

    kw_fs = assign_font_size(propercase_freq, 100, 28) #keyword_font_size

    kw_max_dim = max_dimensions(kw_fs)


    words = []
    for kw, d in kw_max_dim.items():
        words.append(Word(kw, {"width": d[0], "height": d[1]}, kw_fs[kw], kw_cluster[kw]))

    return words

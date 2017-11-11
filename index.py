#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
import numpy as np
from sklearn.cluster import AffinityPropagation


def stop_word_stripper(line):
    """
    docstring here
        :param line: nlp object of spacy
        :return: string without stopwords nor punctuation nor symbols
    """
    stop_words = [w.strip('\n').lower() for w in open('stop_words.txt').readlines()]
    pos_stopper = ['PUNCT', 'SYM']

    return ' '.join([token.text for token in line if str(token).lower() not in stop_words and token.pos_  not in pos_stopper])

'''
PRE-PROCESSING
'''
def pre_processing():
    # load nlp
    nlp = spacy.load('en_core_web_md')

    # load text data
    docs = [x.strip('\n') for x in open('answers.txt').readlines()]
    
    # Strip stop words and create a copy
    stripped_docs = [] #spacy object
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

    corpus = nlp(' '.join(copy_docs))

    MIN_CHARACTERS = 3

for token in corpus:
    if len(token.lemma_) > MIN_CHARACTERS:
        if token.lemma_ in kw_freq:
            kw_freq[token.lemma_] += 1
        else:
            kw_freq[token.lemma_] = 1

print(kw_freq)

# spellcheck keywords: 
from hunspell import Hunspell
h = Hunspell();
spell_checked_kw_freq = {}
for kw, freq in kw_freq.items():
    if freq == 1:
        found = False
        c_kws = h.suggest(kw) 
        for c in c_kws:
            if c in kw_freq:
                spell_checked_kw_freq[c] = kw_freq[c] + 1
                found = True
                break
        if not found:
            spell_checked_kw_freq[c_kws[0]] = 1
print(spell_checked_kw_freq) 

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

print(propercase_freq)
        

# semantic k means clustering

glove_vectors = []
labels_array = []

for kw, count in propercase_freq.items():
    labels_array.append(kw)
    glove_vectors.append(nlp(kw)[0].vector)

print(np.array(glove_vectors).shape, labels_array)


# AffinityPropagation clustering 

AffinityPropagation_model = AffinityPropagation()
AffinityPropagation_model.fit(glove_vectors)

cluster_labels    = AffinityPropagation_model.labels_

clusters = {}
kw_cluster = {}
for i in range(len(labels_array)):
    if cluster_labels[i] not in clusters:
        clusters[cluster_labels[i]] = [labels_array[i]]
    else:
        clusters[cluster_labels[i]].append(labels_array[i])
    kw_cluster[labels_array[i]] = cluster_labels[i]

print (kw_cluster)


#distance matrix (len(cluster_labels)^2)

from scipy import spatial

n = len(labels_array)

distance_matrix = np.zeros([n, n])

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = spatial.distance.cosine(glove_vectors[i], glove_vectors[j])


# assign max font size

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
    
    return (label_fs)
        
kw_fs = assign_font_size(propercase_freq, 80, 30) #keyword_font_size
print(kw_fs)


def max_dimensions(kw_fs):
    kw_dimensions = {}
    for kw, fs in kw_fs.items():
        kw_dimensions[kw] = (int(0.7*len(kw)*fs), fs) #x, y (i.e. width, height)
    return kw_dimensions

kw_max_dim = max_dimensions(kw_fs)
print(kw_max_dim)



words = []
for kw, d in kw_max_dim.items():
    words.append(Word(kw, {"width": d[0], "height": d[1]}, kw_fs[kw], kw_cluster[kw]))

cloud = Cloud(words=words)

cloud.create_cloud_svg()

#cloud.compress()

#print(cloud.canvas)
'''
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

for w in cloud.canvas:
    ax.text(w["x"], w["y"], w["word"], fontsize=w["font_size"]//3)

ax.axis([0, 1920, 0, 1080])

'''
#plt.show()

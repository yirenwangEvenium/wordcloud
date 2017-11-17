import spacy
nlp = spacy.load('en_core_web_md')

from hunspell import Hunspell
h = Hunspell()

import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


from word import Word

class PreProcessing():
    #words are user input per line (e.g. keyword1)
    def __init__(self, stop_words_file = 'stop_words.txt', max_font_size=100, min_font_size=28, min_characters=3, min_words_cluster=50):
        self.corpus = "" #raw data that need to be treated for individual submissions
        self.words = [] # words of each iteration
        self.stop_words = [w.strip('\n').lower() for w in open(stop_words_file).readlines()]
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.min_characters = min_characters
        self.entites_freq = {} # somewhat temp?
        self.words_freq = {} # the dict with everything
        self.min_words_cluster = min_words_cluster
        self.number_of_not_recignized_word = 0
        self.glove_vectors = []
        self.words_info = {} #info being cluster, font_size, dimensions #vectors
        self.cluster_model = AffinityPropagation()

    def add_word(self, words):
        self.corpus = words
        nlp_corpus = nlp(self.corpus)
        self.get_entities(nlp_corpus) # result pushed in self.entites_freq
        self.corpus = self.remove_stopwords(nlp_corpus)
        self.lemmatize(self.spell_check()) # result pushed in self.words
        self.get_freq() # result pushed in self.words_freq
        self.words_freq.update(self.entites_freq)
        self.assign_font_size() # results pushed in self.words_info
        #prepare clustering
        #calculate vectors of new word/words
        new_vectors = self.words_to_vec() # [vec0, ... ] corresponds to self.words [ w1, ... ]
        self.glove_vectors += new_vectors
        
        # check for brands type responses that deson't require clustering
        if self.number_of_not_recignized_word > 0.5*len(self.words_freq):
            for w in self.words:
                self.words_info[w]["cluster"] = 0
        if len(self.words_freq) < self.min_words_cluster:
            self.create_clusters()
        else:
            # add a new word / entity to the cluster 
            # (if there are already plenty of words and well defined clusters)
            for i in range(len(self.words)):
                self.words_info[self.words[i]]["cluster"] = self.cluster_model.predict(new_vectors[i])

        #empty out self.words for next round
        self.words = []
        


    def words_to_vec(self):
        glove_vectors = []
        for word in self.words:
            if nlp(word)[0].vector.any() : #if is it an entity, only get the vector of first word ?
                v = nlp(word)[0].vector
                glove_vectors.append(v)
                self.words_info[word]["vector"] = v
            else: #word not found
                self.number_of_not_recignized_word += 1
                glove_vectors.append(np.array([0]*300))
                self.words_info[word]["vector"] = np.array([0]*300)
        return glove_vectors
    
    def get_entities(self, nlp_corpus):
        # check for entities
        for ent in nlp_corpus.ents:
            self.corpus.replace(ent.text, "", 1)
            self.words.append(ent.text)
            if ent.text in self.entites_freq:
                self.entites_freq[ent.text] += 1
            else:
                self.entites_freq[ent.text] = 1

    def get_freq(self):
        for w in self.words:
            if w in self.words_info:
                self.words_freq[w] += 1
            else:
                self.words_freq[w] = 1

    # Remove stop words, remove words less than min_characters
    def remove_stopwords(self, words):
        pos_stopper = ['PUNCT', 'SYM']
        return ' '.join([token.text for token in words if str(token).lower() not in self.stop_words and token.pos_  not in pos_stopper and len(token.text) > self.min_characters])

    # spell checks corpus
    # only lower case words to avoid spelling checking Brand names
    def spell_check(self):
        ws = []

        for word in self.corpus.split(" "):
            if not h.spell(word):
                ws.append(h.suggest(word)[0]) # can be improved by double checking against existing words for better accuracy
            ws.append(word)
        
        return ws
    
    def lemmatize(self, words):
        '''
        word: nlp token
        '''
        for token in nlp(' '.join(words)): #CRADE ? 
            self.words.append(token.lemma_)

    # create the clusters given words
    def create_clusters(self):
        
        self.cluster_model.fit(self.glove_vectors)
        
        i = 0
        for w in self.words_info:
            self.words_info[w]["cluster"] = self.cluster_model.labels_[i] #[clusterNumber1, ... ] corresponds to [gloveVector1 ... ]
            i += 1
    
    
    # font size gets calculated each time
    def assign_font_size(self):
        sorted_tuples = [(k, self.words_freq[k]) for k in sorted(self.words_freq, key=self.words_freq.get, reverse=True)]
        min_count = sorted_tuples[-1][1]
        max_count = sorted_tuples[0][1]
        # sorted kw_fs in decreasing order of font_size
        for kw, count in sorted_tuples:
            if (max_count - min_count) == 0:
                size = int((self.max_font_size - self.min_font_size) / 2.0 + self.min_font_size)
            else:
                #size = int(self.min_font_size + (self.max_font_size - self.min_font_size) * (count * 1.0 / (max_count - min_count)) ** 0.8)
                size = int((self.max_font_size - self.min_font_size)/(max_count - min_count)*count + self.min_font_size - (self.max_font_size - self.min_font_size)/(max_count - min_count)*min_count)
            if kw in self.words_info:
                self.words_info[kw]["font_size"] = size
            else:
                self.words_info[kw] = { "font_size": size }


t = PreProcessing()
t.add_word("hello airplanes are going to leave soon from the airport. There are many airplanes")
print(t.words_info)
import spacy
nlp = spacy.load('en_core_web_md', parsed=False)

from hunspell import Hunspell
h = Hunspell()

import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from PIL import ImageFont, ImageDraw

import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt


from word import Word

class PreProcessing():
    #words are user input per line (e.g. keyword1)
    def __init__(self, stop_words_file = 'stop_words.txt', negation_stop_words_file = 'negation_stop_words.txt', max_font_size=120, min_font_size=30, min_characters=3, min_words_cluster=300, font="Verdana"):
        self.corpus = [] #raw data that need to be treated for individual submissions
        self.font = font
        self.words = [] # words of each iteration
        self.stop_words = [w.strip('\n').lower() for w in open(stop_words_file).readlines()]
        self.neg_stop_words = [w.strip('\n').lower() for w in open(negation_stop_words_file).readlines()]
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.min_characters = min_characters
        self.entites_freq = {} # somewhat temp?
        self.words_freq = {} # the dict with everything
        self.min_words_cluster = min_words_cluster
        self.number_of_not_recignized_word = 0
        self.words_info = {} #info being cluster, font_size, dimensions #vectors
        self.cluster_model = AffinityPropagation()

    def add_word(self, words):
        self.corpus = words
        nlp_corpus = nlp(words) 
        self.get_entities(nlp_corpus) # result pushed in self.entites_freq
        self.corpus = self.remove_stopwords(nlp(self.corpus))
        self.lemmatize(self.spell_check()) # result pushed in self.words
        if len(self.words) == 0:
            words = [] 
            for w, info in self.words_info.items():
                words.append(Word(word = w, font_size=info["font_size"], size={"width": info["size"][0], "height": info["size"][1] }, cluster=info["cluster"]))
            return words
        self.get_freq() # result pushed in self.words_freq
        self.words_freq.update(self.entites_freq)
        self.assign_font_size() # results pushed in self.words_info
        self.assign_width_height()
        #prepare clustering
        #calculate vectors of new word/words
        self.words_to_vec() # [vec0, ... ] corresponds to self.words [ w1, ... ]
        
        # check for brands type responses that deson't require clustering
        if self.number_of_not_recignized_word + len(self.entites_freq) > 0.4*len(self.words_freq):
            for w in self.words:
                self.words_info['{}{}'.format(w[0].upper(), w[1:].lower())]["cluster"] = 0

        if len(self.words_freq) < self.min_words_cluster:
            self.create_clusters()
        
        else:
            # add a new word / entity to the cluster 
            # (if there are already plenty of words and well defined clusters)
            for i in range(len(self.words)):
                self.words_info['{}{}'.format(self.words[i][0].upper(), self.words[i][1:].lower())]["cluster"] = self.cluster_model.predict([new_vectors[i]])[0]

        #empty out self.words for next round
        self.words = []

        words = [] 
        for w, info in self.words_info.items():
            words.append(Word(word = w, font_size=info["font_size"], size={"width": info["size"][0], "height": info["size"][1] }, cluster=info["cluster"]))
        return words
        

    def words_to_vec(self):
        for word in self.words_info:
            if nlp(word)[0].has_vector : #if is it an entity, only get the vector of first word ?
                v = nlp(word)[0].vector
                self.words_info['{}{}'.format(word[0].upper(), word[1:].lower())]["vector"] = v
            else: #word not found
                self.number_of_not_recignized_word += 1
                self.words_info['{}{}'.format(word[0].upper(), word[1:].lower())]["vector"] = np.array([0]*300)
    
    def get_entities(self, nlp_corpus):
        # check for entities
        for ent in nlp_corpus.ents:
            if len(ent.text.split(' ')) > 1:
                print(self.corpus, ent.text)
                self.corpus=self.corpus.replace(ent.text, "", 1)
                print(self.corpus)
                self.words.append(ent.text)
                if ent.text in self.entites_freq:
                    self.entites_freq[ent.text] += 1
                else:
                    self.entites_freq[ent.text] = 1

        for token in nlp_corpus:
            if len([t for t in token.lefts]) > 0:
                left_token = [t.text for t in token.lefts][0]
                if left_token in self.neg_stop_words :
                    self.words.append(left_token + ' ' + token.text)
                    self.corpus = self.corpus.replace(left_token + ' ' + token.text, '', 1)
                    if left_token + ' ' + token.text in  self.entites_freq:
                        self.entites_freq[left_token + ' ' + token.text] += 1
                    else:
                        self.entites_freq[left_token + ' ' + token.text] = 1
        #bi grams 
        #count pairs of words 
        self.bi_freq = {}
        new_pairs = []
        words = self.corpus.split(' ')
        for i in range(len( words ) - 1):
            bi_w = words[i] + ' ' + words[i+1]
            new_pairs.append(bi_w)
            if bi_w in self.bi_freq:
                self.bi_freq[bi_w] += 1
            else:
                self.bi_freq[bi_w] = 1

        for bi_w, f in self.bi_freq.items():
            if f >= 2 and bi_w in new_pairs:
                self.corpus = self.corpus.replace(bi_w, '', 1)
                if bi_w in self.entites_freq:
                    self.entites_freq[bi_w] += 1
                else:
                    self.entites_freq[bi_w] = 1
            

    def check_in_words_info(self, w):
        if len(self.words_info) == 0:
            return False, None, None, None

        for word in self.words_info:
            # if new word exists within a word that already exists
            if w.lower() in word.lower():
                return True, word, None, None
            
            # if new word is a bigger version of one that already exists 
            elif word.lower() in w.lower():
                return False, w, self.words_freq[word] + 1, word
            
        return False, None, None, None

    def get_freq(self):
        for i in range(len(self.words)):
            w = self.words[i]
            exists, word, freq, prev_word = self.check_in_words_info('{}{}'.format(w[0].upper(), w[1:].lower()))
            if exists:
                # word that is already in dict updated to a higher freq
                self.words = self.words[:i] + [word] + self.words[i+1:]
                self.words_freq[word] += 1
            elif word is not None :
                # replace the word that is in the dict to new word with new freq
                self.words = self.words[:i] + [word] + self.words[i+1:]
                # replacement in words_freq
                self.words_freq[word] = freq
                del self.words_freq[prev_word]
                # replacement in words info 
                self.words_info[word] = self.words_info[prev_word]
                del self.words_info[prev_word]
            else:
                self.words_freq['{}{}'.format(w[0].upper(), w[1:].lower())] = 1

        '''
        for entity, freq in self.entites_freq.items():
            
            w = entity
            exists, word, freq1 = self.check_in_words_info('{}{}'.format(w[0].upper(), w[1:].lower()))
            if exists:
                self.words_freq[word] += 1
            elif word is not None :
                # replace the word in self.words
                self.words_freq[word] = freq1 + freq - 1
            else:
                self.words_freq[w] = freq
        '''

    # Remove stop words, remove words less than min_characters
    def remove_stopwords(self, words):
        pos_stopper = ['PUNCT', 'SYM']
        return ' '.join([token.text for token in words if str(token).lower() not in self.stop_words and token.pos_  not in pos_stopper and len(token.text) > self.min_characters])

    # spell checks corpus
    # only lower case words to avoid spelling checking Brand names
    def spell_check(self):
        ws = []

        for word in self.corpus.split(" "):
            
            if not h.spell(word) and word != "" and word.lower() == word:
                try: 
                    ws.append(h.suggest(word)[0]) # can be improved by double checking against existing words for better accuracy
                except:
                    ws.append(word)
            else:
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
        if len(self.words_info) < 5:
            for w in self.words_info:
                self.words_info[w]["cluster"] = 0
        else:
            vectors = []
            for w, info in self.words_info.items():
                vectors.append(info["vector"])
            self.cluster_model.fit(vectors)
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
        for w, count in sorted_tuples:
            if (max_count - min_count) == 0:
                size = int((self.max_font_size - self.min_font_size) / 2.0 + self.min_font_size)
            else:
                #size = int(self.min_font_size + (self.max_font_size - self.min_font_size) * (count * 1.0 / (max_count - min_count)) ** 0.8)
                size = int((self.max_font_size - self.min_font_size)/(max_count - min_count)*count + self.min_font_size - (self.max_font_size - self.min_font_size)/(max_count - min_count)*min_count)
            if '{}{}'.format(w[0].upper(), w[1:].lower()) in self.words_info:
                self.words_info['{}{}'.format(w[0].upper(), w[1:].lower())]["font_size"] = size
            else:
                self.words_info['{}{}'.format(w[0].upper(), w[1:].lower())] = { "font_size": size }

    def assign_width_height(self):
        for w, info in self.words_info.items():
            font = self.fs_size[info["font_size"]]
            self.words_info[w]["size"] = (ImageFont.ImageFont.getsize(font, w)[0][0], ImageFont.ImageFont.getsize(font, w)[0][1]) #x, y (i.e. width, height)
            for i in w.lower()[1:]:
                if i in ['p', 'g', 'j', 'q', 'y', 'z']:
                    self.words_info[w]["size"] = (ImageFont.ImageFont.getsize(font, w)[0][0], ImageFont.ImageFont.getsize(font, w)[0][1]*1.2) #x, y (i.e. width, height)
                    break
            
    def set_font_size_to_size(self):
        self.fs_size = {}
        for i in range(self.min_font_size - 1, self.max_font_size + 1):
            font = ImageFont.truetype(font=self.font, size=i)
            self.fs_size[i] = font

    def create_force_cloud(self):
        G = nx.Graph()

        for w, info in self.words_info.items():
            for w1, info1 in self.words_info.items():
                if w != w1:
                    if not G.has_edge(w, w1):
                        if info["cluster"] == info1["cluster"]:
                            weight = 5 - spatial.distance.cosine(info["vector"], info1["vector"]) + 5
                            if weight == weight:
                                G.add_edge(w, w1, weight=weight)
                            else:
                                G.add_edge(w, w1, weight=0.1)
                        else:
                            weight = 5 - spatial.distance.cosine(info["vector"], info1["vector"])
                            if weight == weight:
                                G.add_edge(w, w1, weight=weight)
                            else:
                                G.add_edge(w, w1, weight=0.1)

        forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)
        positions = {}
        for p, pos in forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100).items():
            positions[p] = (min(pos[0], 10000), min(pos[1], 10000) )
            print(pos)
        
        edgelist=[]

        nx.draw_networkx(G, positions, edgelist=edgelist, cmap=plt.get_cmap('jet'), node_size=50, with_labels=True, vmin=0.,vmax=0.6)
        plt.show()

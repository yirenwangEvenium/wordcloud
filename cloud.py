#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from math import cos
from math import sin

class Cloud:
    def __init__(self, words=[], canvas_size={"x": 1920, "y": 1080}, filename='clouds.html'):
        self.words = words
        self.canvas = [] #{word, font_size, x, y, width, height, color, cluster} <== color to be added
        self.canvas_size = canvas_size
        self.clusters = self.generate_clusters() # {0 : cluster0, 1 : cluster1, ...etc}
        self.filename = filename
        self.colors = ["#5F0B2B","#D11638", "#F08801", "#FACE00", "#ADA20B", "#D4D639", "#EF6D3E", "#90A6BF", "#F2AEB4", "#AA61AE", "#FBC6CE"]
        self.positions = []
        
    def generate_clusters(self):
        clusters = {}
        for w in self.words:
            if w.cluster in clusters:
                clusters[w.cluster].append(w)
            else:
                clusters[w.cluster] = [w]
        return clusters
        
    def create_cloud_svg(self):
        cl_size = {}
        for c, words in self.clusters.items():
            cl_size[c] = len(words)
        sorted_clusters = sorted(cl_size, key=cl_size.get)[::-1]
        
        start_position = { "x": 1920//2, "y": 1080//3 }
        
        for c in sorted_clusters:
            words = self.clusters[c]
            self.positions = self.spiral(start_position)
            for w in words:
                new_position = self.add_word_to_cloud(w) 
                
            start_position = new_position
            
        f = open(self.filename, 'w')
        f.write('<svg viewbox="0 0 1920 1080">')
        for w in self.canvas:
            f.write('<text x="{}" y="{}" font-family="Verdana" font-size="{}" stroke="none" fill="{}">'.format(w["x"], w["y"], w["font_size"], w["color"]))
            f.write(w["word"])
            f.write('</text>\n')
        f.write('</svg>')
        f.close()


    def add_word_to_cloud(self, word): # word class Word        
        """
        docstring here
            :param word: Class Word
            :return:
        """
        for p in self.positions:
            if self.verify_overlap( word, p):
                self.positions.remove(p)
            else:
                self.canvas.append({
                    "word": word.word,
                    "x": p["x"],
                    "y": p["y"],
                    "width": word.width,
                    "height": word.height,
                    "font_size": word.font_size,
                    "color": self.colors[word.cluster],
                    "cluster": word.cluster
                })
                self.positions.remove(p)
                return p


    def rect_intersection(self, r1, r2):
        # you need x,y width height for each rectangle (word)
        # r1 x, y, width, height
        # p1--------
        #  |        |
        #  |        |
        # (x,y)-----p2


        p1 = {}
        p1["x"] = r1["x"]
        p1["y"] = r1["y"] - r1["height"]

        p2 = {}
        p2["x"] = r1["x"] + r1["width"]
        p2["y"] = r1["y"]

        p3 = {}
        p3["x"] = r2["x"]
        p3["y"] = r2["y"] - r2["height"]

        p4 = {}
        p4["x"] = r2["x"] + r2["width"]
        p4["y"] = r2["y"]

        return not(p2["y"] < p3["y"] or p1["y"] > p4["y"] or p2["x"] < p3["x"] or p1["x"] > p4["x"])

    
    def verify_overlap(self, word, position): # true if overlaps, false if not
        new_rect = {
            "x": position["x"],
            "y": position["y"],
            "width": word.width,
            "height": word.height
        }
        for filled_rect in self.canvas:
            if self.rect_intersection(filled_rect, new_rect):
                return True
        #verify out of bound of rectangle:
        if new_rect["x"] < 0 or new_rect["x"] + new_rect["width"] > 1920 or new_rect["y"] > 1080 or new_rect["y"]- new_rect["height"] < 0:
            return True
        return False
    

    def spiral(self, start_point): # returns an [] with positions to test 
        points = [start_point]
        # x = (a + b*theta)cos(theta)
        # y = (a + b*theta)sin(theta)

        # b = a final - a ini / 2 pi n  n=number of turns
        a_ini = 0
        # a_final = self.canvas_size["x"]*len(self.clusters[cluster])/len(self.words) #spiral radius 
        a_final = self.canvas_size["x"] #spiral radius 

        b = (a_final - a_ini)/(2*3.14159*(self.canvas_size["y"]/10))

        thetas = [ (self.canvas_size["y"]/10 * 2)/1000 *x for x in range(1000)]
        for i in thetas: #1000 points
            x = ( a_ini + b*i + cos(i)*b/10)*cos(i) + start_point["x"]
            y = ( a_ini + b*i + cos(i)*b/10)*sin(i) + start_point["y"]
            points.append({"x": x, "y": y})

        return points
    
    '''
    def choose_cluster_start(self):
        start_points = {}
        start_point = {}
        r = 0
        for i in range(len(self.clusters)):
            c = self.clusters[i]
            n = len(c)
            
            H = self.canvas_size["y"] #total height
            L = self.canvas_size["x"] #total length
            
            if i%2 == 0:
                y = random.randint(int(0.1*H), int(0.55*H))
            else:
                y = random.randint(int(0.55*H), int(0.9*H))
            x = random.randint(int(r*L), min(int((r+len(c)/len(self.words))*L), int(L*0.90)))
            
            r = min(0.85, r + len(c)/len(self.words))
            start_points[c[0].cluster] = {
                "x": x,
                "y": y
            }
        return start_points
    
    def compress(self):
        # pull words towards the one zith the most occurence
        # create line 
        # test positions along that line 
        sizes = []
        for w in self.canvas:
            sizes.append(w["font_size"])
        central_word = self.canvas[sizes.index(max(sizes))]
        
        for w in self.canvas:
            if w["cluster"] != central_word["cluster"]:
                # sort tham by distance 
                pos_central_word = np.array([central_word["x"], central_word["y"]])
                pos_w = np.array([w["x"], w["y"]])
                dist = numpy.sqrt(numpy.sum((pos_central_word - pos_w)**2))
                
                # draw line 
                # inch closer 
                coeff = central_word["y"] - w["y"] / central_word["x"] - w["x"]
                coordiantes = [{"x": central_word["x"] + (central_word["x"] - w["x"])/100 * i, "y": central_word["y"] + coeff* (central_word["x"] - w["x"])/100 * i } for i in range(100)]
                for c in coordinates:
                    for word in self.words:
                        if self.verify_overlap(word, c):
                            break
    '''             
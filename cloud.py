import random
from math import cos
from math import sin


class Cloud:
    def __init__(self, color, words=[] , canvas_size={"x": 1920, "y": 1080}, filename='clouds.html', spiral_size = 15, min_font_size=28, max_font_size=100):
        self.color = color
        self.words = words
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.spiral_size = spiral_size
        self.canvas = [] #{word, font_size, x, y, width, height, color, cluster} <== color to be added
        self.canvas_size = canvas_size
        self.clusters = self.generate_clusters() # {0 : cluster0, 1 : cluster1, ...etc}
        self.filename = filename
        self.colors = ["#FE4747", "#FDA47C", "#FBD094", "#D9CC8F", "#A8B47D", "#AFE457", "#9C519B", "#EF93A4", "#BFBED9", "#F6C04E", "#FCEBC2", "#EDE574", "#EDE573", "#EDE524", "#EDF574", "#EDF574", "#EDF574", "#EDF574", "#EDF574", "#EDF574", "#EDF574"]
        self.positions = []
        self.sorted_clusters = []
        self.previously_check_fail = None
    
    def generate_clusters(self):
        '''
        gather the words in the appropriate clusters
        '''
        clusters = {}
        for w in self.words:
            if w.cluster in clusters:
                clusters[w.cluster].append(w)
            else:
                clusters[w.cluster] = [w]
        
        return clusters
    
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
    '''
        
    def create_cloud(self):
        
        # sort by cluster size
        # sort by max font-size
        cl_size = {}
        for c, words in self.clusters.items():
            if len(words) < 4:
                avg_size = sum([w.font_size for w in words])//len(words)
            else:
                avg_size = sum(sorted([w.font_size for w in words])[::-1][:4])/4
            cl_size[c] = avg_size*3 - len(words)
        self.sorted_clusters = sorted(cl_size, key=cl_size.get)[::-1]
        
        start_position = { "x": self.canvas_size["x"]//2, "y": self.canvas_size["y"]//2 }
        
        for i in range(len(self.sorted_clusters)):
            c = self.sorted_clusters[i]
            words = self.clusters[c]
            self.positions = self.spiral(start_position)
            
            for w in words:
                if len(self.positions) == 0:
                    break
                new_position = self.add_word_to_cloud(w, i%2 == 0) 
            
            max_left_cloud = min([c["x"] for c in self.canvas])
            max_right_cloud = max([c["x"] for c in self.canvas])
            shift = 30
            if i%2 == 0:
                if new_position["x"] < self.canvas_size["x"]//2: 
                    start_position = { "x" : min(self.canvas_size["x"]//2 + new_position["x"], max_right_cloud + shift), "y": new_position["y"] }
                if new_position["x"] > self.canvas_size["x"]//2: 
                 start_position = { "x" : max(self.canvas_size["x"] - new_position["x"], max_left_cloud - shift), "y": new_position["y"] }
            else:
                start_position = new_position
        
        self.center_cloud()
        
    def draw_cloud_to_svg(self):
        f = open(self.filename, 'w')
        f.write('<svg viewbox="0 0 {} {}" style="background: black">'.format(self.canvas_size["x"], self.canvas_size["y"]))
        for w in self.canvas:
           

            #f.write(' <rect x="{}" y="{}" width="{}" height="{}"/>'.format( w["x"], w["y"], w["width"], w["height"]))
            f.write('<text x="{}" y="{}" font-family="Verdana" font-size="{}" fill="{}">'.format(w["x"], w["y"], w["font_size"], w["color"]))
            f.write(w["word"])
            f.write('</text>\n')
        f.write('</svg>')
        f.close()
        
        
    def add_word_to_cloud(self, word, moved): # word class Word
        center = {"x": self.canvas_size["x"] // 2, "y": self.canvas_size["y"] // 2}
        last_position = self.positions[-1]
        for p in self.positions:
            if p["x"] < center["x"] and not moved:
                if not self.verify_overlap( word, {"x": p["x"] - word.width, "y": p["y"]} ):
                    self.canvas.append({
                        "word": word.word,
                        "x": p["x"] - word.width,
                        "y": p["y"],
                        "width": word.width,
                        "height": word.height,
                        "font_size": word.font_size,
                        "color": self.color.choose_color(word.font_size, self.sorted_clusters.index(word.cluster)),
                        "cluster": word.cluster
                    })
                    self.positions.remove(p)
                    return p
                self.positions.remove(p)
            else:
                if not self.verify_overlap( word, {"x": p["x"], "y": p["y"]} ):
                    self.canvas.append({
                        "word": word.word,
                        "x": p["x"],
                        "y": p["y"],
                        "width": word.width,
                        "height": word.height,
                        "font_size": word.font_size,
                        "color": self.color.choose_color(word.font_size, self.sorted_clusters.index(word.cluster)),
                        "cluster": word.cluster
                    })
                    self.positions.remove(p)
                    return p
                self.positions.remove(p)
        if len(self.positions) == 0:
            return last_position
        return self.positions[-1]
            

    def rect_intersection(self, r1, r2):
        
        p1 = {}
        p1["x"] = r1["x"]*0.99
        p1["y"] = (r1["y"] - r1["height"])*0.99

        p2 = {}
        p2["x"] = (r1["x"] + r1["width"])*1.01
        p2["y"] = r1["y"]*1.01

        p3 = {}
        p3["x"] = r2["x"]*0.99
        p3["y"] = (r2["y"] - r2["height"])*0.99

        p4 = {}
        p4["x"] = (r2["x"] + r2["width"])*1.01
        p4["y"] = r2["y"]*1.01

        return not(p2["y"] < p3["y"] or p1["y"] > p4["y"] or p2["x"] < p3["x"] or p1["x"] > p4["x"])

    
    def verify_overlap(self, word, position): # true if overlaps, false if not
        new_rect = {
            "x": position["x"],
            "y": position["y"],
            "width": word.width,
            "height": word.height
        }

        # check last failed position for faster iterations
        if self.previously_check_fail is not None:
            if self.rect_intersection(self.previously_check_fail, new_rect):
                return True
                
        for filled_rect in self.canvas:
            if self.rect_intersection(filled_rect, new_rect):
                self.previously_check_fail = filled_rect
                return True
        #verify out of bound of rectangle:
        if new_rect["x"] < 10 or new_rect["x"] + new_rect["width"] > self.canvas_size["x"]*0.99 or new_rect["y"] > self.canvas_size["y"]*0.99 or new_rect["y"]- new_rect["height"] < 10:
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

        b = (a_final - a_ini)/(2*3.14159*(self.canvas_size["x"]/self.spiral_size))

        thetas = [ (self.canvas_size["y"]/10 * 2)/1500 *x for x in range(1500)]
        for i in thetas: #1000 points
            x = ( a_ini + b*i + cos(i)*b/10)*cos(i) + start_point["x"]
            y = ( a_ini + b*i + cos(i)*b/10)*sin(i) + start_point["y"]
            points.append({"x": x, "y": y})
        return points
    
    def center_cloud(self):
        xs = [c["x"] for c in self.canvas]
        ys = [c["y"] for c in self.canvas]
        
        x_min = min(xs)
        x_max = max(xs) + self.canvas[xs.index(max(xs))]["width"]
        
        y_min = min(ys)
        y_max = max(ys)
        
        shift_x = x_min - (self.canvas_size["x"] - (x_max - x_min))//2
        shift_y = y_min - (self.canvas_size["y"] - (y_max - y_min))//2
        
        for c in self.canvas:
            c["x"] -= shift_x
            c["y"] -= shift_y
        
        
    '''
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


import os
import time

from cloud import Cloud
from pre import PreProcessing
from colors import Colors

# testing 
start = time.time()
text = PreProcessing()

for line in [line.strip('\n') for line in open(os.path.join('data/', 'real7.txt')).readlines()]:
    words = text.add_word(line)
print([w.word for w in words])
print([w.font_size for w in words])
print([w.cluster for w in words])
pre = time.time()
pre_time = time.time() - start 

color = Colors()

cloud = Cloud(words=words, color = color, filename='clouds/{}.html'.format('real7.txt'.replace('.txt', '')), spiral_size=15)
cloud.create_cloud()
cloud.draw_cloud_to_svg()

create_time = time.time() - pre 
total_time = time.time() - start 
print(pre_time, create_time, total_time)
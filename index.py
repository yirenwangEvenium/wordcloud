import os
import time
import re

from cloud import Cloud
from pre import PreProcessing
from colors import Colors


color = Colors()

# testing 
for f in os.listdir('data'):
    text = PreProcessing()

    #set font_sizes
    text.set_font_size_to_size()
    
    start = time.time()

    for line in [line.strip('\n') for line in open(os.path.join('data/', f)).readlines()]:
        words = text.add_word(re.sub('[^0-9a-zA-Z -]+', ' ', line))
        #words = text.add_word(line)

    pre = time.time()
    pre_time = time.time() - start 

    print(text.entites_freq)
    cloud = Cloud(words=words, color = color, filename='clouds/{}.html'.format(f.replace('.txt', '')), spiral_size=15)
    cloud.create_cloud()
    cloud.draw_cloud_to_svg()

    
    create_time = time.time() - pre 
    total_time = time.time() - start 
    print(f, pre_time, create_time, total_time)

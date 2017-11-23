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
        a = time.time()
        words = text.add_word(re.sub('[^0-9a-zA-Z -]+', ' ', line))
        #words = text.add_word(line)

        pre = time.time()
        pre_time = time.time() - a 

        cloud = Cloud(words=words, color = color, filename='clouds/{}.html'.format(f.replace('.txt', '')), spiral_size=15)
        cloud.create_cloud()

        pre_c = time.time()

        for i in range(10):
            cloud.compress()
        cloud.draw_cloud_to_svg()

        compress_time = time.time() - pre_c
        
        create_time = time.time() - pre 
        total_time = time.time() - a 
        print(f, "preprocesssing" , pre_time, "creation", create_time, "compress", compress_time, "total", total_time)

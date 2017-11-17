import os
import time

from cloud import Cloud
from preprocessing import pre_processing
from colors import Colors

# testing 
for f in os.listdir('data'):
    start = time.time()

    words = pre_processing(f)
    print([w.word for w in words])
    pre = time.time()
    pre_time = time.time() - start 

    color = Colors()

    cloud = Cloud(words=words, color = color, filename='clouds/{}.html'.format(f.replace('.txt', '')), spiral_size=15)
    cloud.create_cloud()
    cloud.draw_cloud_to_svg()

    create_time = time.time() - pre 
    total_time = time.time() - start 
    print(f, pre_time, create_time, total_time)
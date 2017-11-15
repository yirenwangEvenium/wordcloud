import os
import time

from cloud import Cloud
from preprocessing import pre_processing


# testing 
for f in os.listdir('data'):
    start = time.time()

    words = pre_processing(f)
    pre = time.time()
    pre_time = time.time() - start 

    cloud = Cloud(words=words, filename='clouds/{}.html'.format(f.replace('.txt', '')), spiral_size=20)
    cloud.create_cloud()
    cloud.draw_cloud_to_svg()

    create_time = time.time() - pre 
    total_time = time.time() - start 
    print(f, pre_time, create_time, total_time)
import os

from cloud import Cloud
from preprocessing import pre_processing

for f in os.listdir('data'):
    words = pre_processing(f)
    cloud = Cloud(words=words, filename='{}.html'.format(f.replace('.txt', '')))
    cloud.create_cloud()
    cloud.draw_cloud_to_svg()

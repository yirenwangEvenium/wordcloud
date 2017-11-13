from cloud import Cloud
from preprocessing import pre_processing

words = pre_processing('real3.txt')
cloud = Cloud(words=words)

cloud.create_cloud()

cloud.draw_cloud_to_svg()

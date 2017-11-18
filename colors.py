import colorsys
import random 

class Colors:
    def __init__(self, min_font_size = 28, max_font_size=100):
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.base_colors = self.create_base_colours()
    
    def create_base_colours(self):
        hsv0 = (0, 1, 1)
        colors = []
        for i in range(1, 50): #50 base colours
            hsv = (i*62%360/360, 1, 1)
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            hex = self.rgb_to_hex(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex)
        return colors

    def hex_to_rgb(self,value):
        """Return (red, green, blue) for the color given as #rrggbb."""
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_hex(self, red, green, blue):
        """Return color as #rrggbb for the given color values."""
        return '#%02x%02x%02x' % (red, green, blue)

    def choose_color(self, font_size, cluster_priority):
        if cluster_priority < len(self.base_colors):
            base_color = self.base_colors[cluster_priority]
        else:
            base_color = self.base_colors[-1]
        saturation = font_size/self.max_font_size
        rgb = self.hex_to_rgb(base_color)
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        rgb2 = colorsys.hsv_to_rgb(hsv[0], saturation, hsv[2])
        return self.rgb_to_hex(int(rgb2[0]*255), int(rgb2[1]*255), int(rgb2[2]*255))


c = Colors()
print(c.base_colors)
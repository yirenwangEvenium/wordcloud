import colorsys

class Colors:
    def __init__(self, min_font_size, max_font_size):
        base_colors = [] #brightest to darkest TODO with this chart: http://www.december.com/html/spec/colorhslhex10.html
    
    def hex_to_rgb(value):
        """Return (red, green, blue) for the color given as #rrggbb."""
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_hex(red, green, blue):
        """Return color as #rrggbb for the given color values."""
        return '#%02x%02x%02x' % (red, green, blue)

    def chooseColor(self, font_size, cluster_priority):
        base_color = self.base_colors[cluster_priority]
        saturation = font_size/(max_font_size - min_font_size)
        rgb = hex_to_rgb(base_color)
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        rgb2 = colorsys.hsv_to_rgb(hsv[0], saturation, hsv[2])
        return rgb_to_hex(rgb2[0], rgb2[1], rgb2[2])

    


#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Word:
    def __init__(self, word, size, font_size, cluster):
        self.word = word
        self.width = size["width"] #{width, height}
        self.height = size["height"]
        self.font_size = font_size
        self.cluster = cluster

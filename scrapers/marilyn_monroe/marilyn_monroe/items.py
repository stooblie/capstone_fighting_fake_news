# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MarilynMonroeItem(scrapy.Item):
    title = scrapy.Field()
    base_url = scrapy.Field()
    url = scrapy.Field()
    text = scrapy.Field()
    image_hash_ids = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()

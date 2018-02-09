# import the necessary packages
from harvey.items import HarveyItem
from scrapy.utils.response import get_base_url
from scrapy import Selector
import scrapy
from urllib.parse import urljoin
import boto3
import botocore
import hashlib
from collections import defaultdict

class HarveySpider(scrapy.Spider):
	name = "harvey-spider"
	start_urls = ['http://vidloops.com/gators-coffins-and-snakes-what-60-inches-of-r-uxU_EFHJObA',
	'https://nypost.com/2016/10/09/floridians-cleaning-up-hurricane-matthew-damage-surrounded-by-gators/',
	'http://www.dailypress.com/topic/science/scientific-research/animal-research/13012000-topic.html',
	'https://www.thealternativedaily.com/how-a-hurricane-shakes-the-animal-kingdom/',
	'https://www.thesun.co.uk/news/4448508/miami-sharks-flooded-street-hurricane-irma-video/',
	'https://www.theguardian.com/us-news/2017/sep/03/in-houston-wayward-alligators-look-to-return-home-too',
	'http://www.devityhotel.com/music/hurricane-harvey-warning-beware-of-alligators.html',
	'https://niketalk.com/threads/hurricane-harvey-texas-nters-stay-safe.665314/page-28',
	'http://newschannel9.com/news/nation-world/receding-flood-waters-from-harvey-turning-up-unexpected-wildlife',
	'https://www.aol.com/weather/tag/hurricane-harvey/pg-3/',
	'http://www.gizmodo.in/news/harvey-flooding-dredges-up-gators-fire-ants-and-bats/articleshow/60262039.cms',
	'http://blog.wildlifejournalist.com/category/gulf-of-mexico/',
	'https://www.leighelena.com/collections/all/sale?page=2',
	'https://www.pinterest.co.uk/?show_error=true',
	'http://www.lastminutestuff.com/content/LA-police-find-house-full-of-venomous-snakes-and/4308399.html',
	'https://www.pinterest.com.au/pin/487936940871003880/',
	'http://sportingclassicsdaily.com/author/sportingclassics_daily/page/14/',
	'https://www.msn.com/en-ca/news/world/this-homeowner-found-a-10-foot-gator-in-his-flooded-home-near-houston/ar-AAr7Hw2?li=AAggFp5',
	'http://www.flave.online/topic/warning-alligators-at-large-following-harvey-flooding-breaking-news',
	'http://www.cljnews.com/',
	'http://sciencehours.com/nature/this-nightmare-shark-with-a-snake-head-and-300-teeth-is-absolutely-horrifying-2/',
	'https://www.pinterest.com/pin/575334921138206749/',
	'http://www.ksdk.com/article/news/nation-now/this-sea-creature-found-after-hurricane-harvey-is-pretty-gnarly/465-bf789410-3d3e-4ad8-95c5-4cecaff69d44',
	]

	def parse(self, response):
		#Gather links from the page to parse
		links = set(response.css('a::attr(href)').extract())
		links = links - set(response.css('a[class="image"]::attr(href)').extract())

		for lnk in links:
			if 'http' not in lnk: continue
			yield scrapy.Request(lnk, self.parse_page)

		# for i in links:
		# 	if 'http' not in i:
		# 		links.remove(i)
		# 		links.add('https:' + i)
		# 	try:
		# 		yield scrapy.Request(i, self.parse_page)
		# 	except:
		# 		print('Request is breaking on link:', i, type(i))

	def parse_page(self, response):
		# Given a page response, parse all the images and grab the title
		sel = Selector(response=response)

		title = response.css('title::text').extract()
		text = ''.join(sel.select("//body//text()").extract()).strip()
		url = response.request.url
		base = get_base_url(response)

		image_urls = [urljoin(response.url, src) for src in response.xpath('//img/@src').extract()]
		image_hashes = defaultdict()
		for img in image_urls: image_hashes[hashlib.sha1(img.encode()).hexdigest()] = img

		yield HarveyItem(title=title, base_url=base, url=url, text=text, image_hash_ids=image_hashes, image_urls=image_urls)
		#
		# # extract the 'Next' link from the pagination, load it, and
		# # parse it
		# next = response.css("div.pages").xpath("a[contains(., 'Next')]")
		# yield scrapy.Request(next.xpath("@href").extract_first(), self.parse_page_images)

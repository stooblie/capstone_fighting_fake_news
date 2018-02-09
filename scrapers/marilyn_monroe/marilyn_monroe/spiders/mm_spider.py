# import the necessary packages
from marilyn_monroe.items import MarilynMonroeItem
from scrapy.utils.response import get_base_url
from scrapy import Selector
import scrapy
from urllib.parse import urljoin
import boto3
import botocore
import hashlib
from collections import defaultdict

class MarilynMonroeSpider(scrapy.Spider):
	name = "mm-spider"
	start_urls = [
	'https://en.wikipedia.org/wiki/Marilyn_Monroe',
	'http://www.imdb.com/name/nm0000054/',
	'https://www.vanityfair.com/news/2008/10/marilyn200810',
	'https://www.nytimes.com/2017/01/13/nyregion/marilyn-monroe-skirt-blowing-new-york-film.html?mtrref=www.google.com&gwh=B61D11CC02CE71F2AA457D376FCF70C3&gwt=pay',
	'http://www.pbs.org/wnet/americanmasters/marilyn-monroe-biography/61/',
	'https://www.biography.com/people/marilyn-monroe-9412123',
	'https://marilynmonroe.com/',
	'https://www.vanityfair.com/culture/2010/11/marilyn-monroe-201011',
	'https://www.rottentomatoes.com/celebrity/marilyn_monroe/',
	'http://www.telegraph.co.uk/films/2016/06/01/50-things-you-didnt-know-about-marilyn-monroe/',
	'http://www.tmz.com/person/marilyn-monroe/',
	'http://people.com/tag/marilyn-monroe/',
	'https://www.britannica.com/biography/Marilyn-Monroe',
	'http://www.nydailynews.com/entertainment/marilyn-monroe-50th-anniversary-death-gallery-1.1128658?pmSlide=1.1128605',
	'http://www.dailymail.co.uk/home/event/article-4903108/Marilyn-Monroe-ve-never-seen-before.html',
	'http://www.youmustrememberthispodcast.com/episodes/2017/3/13/marilyn-monroe-the-persona-dead-blondes-episode-7',
	'https://www.pinterest.com/explore/marilyn-monroe/',
	'https://www.popsugar.com/celebrity/JFK-Marilyn-Monroe-Affair-Details-44024770',
	'https://www.brainyquote.com/authors/marilyn_monroe',
	'https://www.famousbirthdays.com/people/marilyn-monroe.html',
	'https://www.moma.org/learn/moma_learning/andy-warhol-gold-marilyn-monroe-1962',
	'https://www.washingtonpost.com/news/arts-and-entertainment/wp/2017/09/28/marilyn-monroe-helped-launch-hugh-hefners-career-but-they-never-even-met/?utm_term=.8b8f9791c49a',
	'https://gizmodo.com/7-viral-photos-of-marilyn-monroe-that-are-totally-fake-1779843804',
	'https://www.pinterest.com/marijaneg/fake-marilyn-monroe-pictures/',
	'https://www.immortalmarilyn.com/even-more-photos-of-marilyn-monroe-that-arent-marilyn-monroe/',
	'https://www.snopes.com/photos/people/jfk-marilyn.asp',
	'https://www.pinterest.com.au/tresamarilyn/marilyn-monroe-photoshopped/',
	'http://themarilynmonroecollection.com/marilyn-monroe-july-1960-pregnancy-true-example-fake-news/',
	'https://hoaxeye.com/2017/03/28/fake-marilyn-monroe-with-a-cat/',
	'https://www.buzzfeed.com/annehelenpetersen/19-pictures-that-will-make-you-think-differently-about-maril?utm_term=.wo93rOq1yL#.avy4158Mgq',
	'https://www.buzzfeed.com/immortalmarilyn/marilyn-monroe-quotesthat-she-never-actually-sa-12dgq?utm_term=.bhoRb0oNLy#.wuyQy2Zrw9',
	'http://www.dailymail.co.uk/news/article-4157738/Never-seen-pictures-pregnant-Marilyn-Monroe.html',
	'http://mashable.com/2017/09/28/marilyn-monroe-hugh-hefner-fake-picture-playboy/#6tib872HeEqS',
	'https://townhall.com/tipsheet/christinerousselle/2017/10/02/marilyn-monroe-hugh-hefner-n2389463',
	'https://www.business2community.com/entertainment/photo-marilyn-monroe-holding-cat-digital-fake-01859664',
	'https://www.boredpanda.com/before-marilyn-monroe-norma-jeane-mortenson-photos/',
	'http://hoaxoffame.tumblr.com/post/111170153346/fake-yes-explanation-marilyn-monroe-and',
	'http://antiviral.gawker.com/forward-or-delete-this-weeks-fake-viral-photos-1705887751',
	'http://www.instyle.com/celebrity/transformations/marilyn-monroes-changing-looks',
	'https://www.reddit.com/r/fatlogic/comments/3puil8/just_popped_up_on_my_facebook_have_they_ever/',
	'https://hellogiggles.com/celebrity/marilyn-monroe-quotes/',
	'https://www.allure.com/story/marilyn-monroe-mystery-plastic-surgery-medical-records',
	'https://www.essence.com/celebrity/alicia-keys-marilyn-monroe-real-bodies-tweet',
	'https://www.biography.com/news/marilyn-monroe-remembered-in-9-ways',
	'http://www.thefrisky.com/2008-05-28/quick-pic-the-sexiest-potentially-fake-marilyn-monroe-image-ever/',
	'http://www.chron.com/entertainment/celebrities/article/Marilyn-Monroe-rare-photos-12234835.php',
	'http://www.slate.com/articles/arts/doonan/2012/01/was_marilyn_monroe_fat_her_secrets_revealed_.html',
	'http://www.alison-jackson.co.uk/mental-images/'
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

		yield MarilynMonroeItem(title=title, base_url=base, url=url, text=text, image_hash_ids=image_hashes, image_urls=image_urls)
		#
		# # extract the 'Next' link from the pagination, load it, and
		# # parse it
		# next = response.css("div.pages").xpath("a[contains(., 'Next')]")
		# yield scrapy.Request(next.xpath("@href").extract_first(), self.parse_page_images)

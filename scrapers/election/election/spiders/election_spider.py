# import the necessary packages
from election.items import ElectionItem
from scrapy.utils.response import get_base_url
from scrapy import Selector
import scrapy
from urllib.parse import urljoin
import boto3
import botocore
import hashlib
from collections import defaultdict
from time import sleep

class ElectionSpider(scrapy.Spider):
	name = "election-spider"
	start_urls = ['https://en.wikipedia.org/wiki/United_States_presidential_election,_2016',
    'https://en.wikipedia.org/wiki/United_States_elections,_2016',
    'https://www.270towin.com/2016_Election/',
    'https://www.nytimes.com/elections/results/president',
    'http://www.cnn.com/election/results',
    'https://www.politico.com/mapdata-2016/2016-election/results/map/president/',
    'https://fivethirtyeight.com/politics/elections/',
    'https://www.cbsnews.com/election-2016/',
    'http://www.people-press.org/2016/07/07/4-top-voting-issues-in-2016-election/',
    'https://www.npr.org/2016/11/08/500686320/did-social-media-ruin-election-2016',
    'https://www.thenation.com/article/how-false-equivalence-is-distorting-the-2016-election-coverage/',
    'https://www.vox.com/policy-and-politics/2018/1/8/16865532/2016-presidential-election-map-xkcd',
    'http://www.pewresearch.org/topics/2016-election/',
    'http://time.com/tag/2016-election/',
    'https://www.theatlantic.com/category/2016-election/',
    'https://www.washingtonpost.com/opinions/the-2016-election-was-not-a-fluke/2017/09/18/b45a8a0e-9cb4-11e7-9083-fbfddf6804c2_story.html?utm_term=.a5b696ecb42b',
    'https://www.britannica.com/topic/United-States-presidential-election-of-2016',
    'http://www.presidency.ucsb.edu/showelection.php?year=2016',
    'http://www.slate.com/articles/news_and_politics/politics/2016/12/_19_lessons_for_political_scientists_from_the_2016_election.html',
    'https://realclearpolitics.com/elections/2016/',
    'http://www.msnbc.com/election-2016',
    'https://www.nbcnews.com/politics/2016-election',
    'https://www.usnews.com/news/the-run-2016/articles/2016-11-14/the-10-closest-states-in-the-2016-election',
    'http://sos.alabama.gov/alabama-votes/voter/election-information/2016',
    'http://www.businessinsider.com/wikileaks-urged-trump-to-contest-the-2016-election-results-if-he-lost-2017-11',
    'https://www.votepinellas.com/Election-Results/2016-Election-Results',
    'http://www.sos.state.mn.us/elections-voting/2016-general-election-results/',
    'http://abcnews.go.com/Politics/key-moments-2016-election/story?id=43289663',
    'https://www.huffingtonpost.com/topic/elections-2016',
    'http://www.politifact.com/truth-o-meter/article/2018/jan/03/more-year-after-2016-election-how-trustworthy-are-/',
    'http://www.aapor.org/Education-Resources/Reports/An-Evaluation-of-2016-Election-Polls-in-the-U-S.aspx',
    'https://www.npr.org/tags/487769724/2016-election',
    'http://vote.sonoma-county.org/content.aspx?sid=1009&id=3332',
    'https://civicyouth.org/quick-facts/2016-election-center/',
    'http://ew.com/books/the-complete-guide-to-books-on-the-2016-election/theres-a-2016-election-book-for-everyone',
    'https://www.theguardian.com/us-news/ng-interactive/2016/nov/08/us-election-2016-results-live-clinton-trump?view=map&type=presidential',
    'http://www.bbc.com/news/world-us-canada-35356941',
    'https://www.aaup.org/news/higher-education-after-2016-election#.Wl4qUZM-fBI',
    'https://www.newyorker.com/tag/2016-election',
    'https://www.usatoday.com/story/news/politics/onpolitics/2017/10/30/whos-who-key-players-investigation-into-russian-interference-2016-election/813596001/',
    'http://www.journalism.org/2016/07/18/election-2016-campaigns-as-a-direct-source-of-news/',
    'https://www.voteosceola.com/en/election-results/2016-election-results/',
    'https://www.snopes.com/2017/09/27/presidential-election-do-over/',
    'http://www.people-press.org/2016/11/21/low-marks-for-major-players-in-2016-election-including-the-winner/',
    'https://www.forbes.com/sites/startswithabang/2016/11/09/the-science-of-error-how-polling-botched-the-2016-election/#6bb5d2ff3795',
    'https://imprimis.hillsdale.edu/2016-election-demise-journalistic-standards/',
    'https://www.nytimes.com/2016/11/09/us/politics/debunk-fake-news-election-day.html',
    'https://www.washingtonpost.com/news/fact-checker/wp/2016/11/16/no-the-viral-image-of-2016-election-results-and-2013-crime-rates-is-not-real/?utm_term=.67957435de4a',
    'https://www.theguardian.com/technology/2016/nov/10/facebook-fake-news-election-conspiracy-theories',
    'https://www.wired.com/2016/12/photos-fuel-spread-fake-news/',
    'http://www.philly.com/philly/news/politics/presidential/facebook-russia-fake-posts-trump-election-clinton-20171006.html',
    'http://www.businessinsider.com/fake-presidential-election-news-viral-facebook-trump-clinton-2016-11',
    'https://firstdraftnews.com/7-types-political-hoaxes-youll-see-us-presidential-election/',
    'https://www.recode.net/2017/9/28/16380544/twitter-facebook-reddit-russia-fake-news-clinton-trump-presidential-election-2016-social-media',
    'https://www.snopes.com/trump-unflattering-image/',
    'http://www.abc.net.au/news/2016-11-14/fake-news-would-have-influenced-us-election-experts-say/8024660',
    'http://www.bbc.com/news/blogs-trending-37945486',
    'http://www.newsweek.com/russia-facebook-ads-fake-news-congressional-hearings-698969',
    'https://www.vox.com/conversations/2017/1/27/14266228/donald-trump-hillary-clinton-fake-news-media-2016-election',
    'https://www.cjr.org/analysis/fake-news-media-election-trump.php',
    'https://www.vox.com/new-money/2016/11/16/13659840/facebook-fake-news-chart',
    'https://www.cbsnews.com/news/paul-horner-fake-news-writer-dead-at-38/',
    'http://www.cnn.com/2017/08/04/politics/election-day-cyber-threat-fbi-monitoring/index.html',
    'http://nymag.com/selectall/2016/11/google-top-election-result-is-fake-news-about-trump.html',
    'https://www.npr.org/sections/alltechconsidered/2017/04/03/522503844/how-russian-twitter-bots-pumped-out-fake-news-during-the-2016-election',
    'https://www.indy100.com/article/no-the-viral-image-of-2016-election-results-and-2013-crime-rates-is-not-real-7421076',
    'https://www.thewrap.com/misinformation-in-2016-a-timeline-of-fake-news-photos/',
    'https://www.huffingtonpost.com/entry/facebook-fake-news-stories-zuckerberg_us_5829f34ee4b0c4b63b0da2ea',
    'http://thehill.com/homenews/media/317646-fake-news-did-not-change-result-of-2016-election-study',
    'http://www.sfgate.com/news/article/Fake-news-stories-that-fooled-liberals-10781580.php',
    'https://www.forbes.com/sites/quora/2016/11/24/did-fake-news-on-facebook-influence-the-outcome-of-the-election/#7e7c47b836e4',
    'http://www.independent.co.uk/news/world/americas/us-politics/facebook-russia-ads-trump-release-fake-accounts-2016-election-copies-a7960356.html',
    'http://www.slate.com/articles/technology/technology/2017/11/here_are_the_facebook_posts_russia_used_to_meddle_in_the_2016_election.html',
    'https://www.politico.com/story/2017/09/06/facebook-ads-russia-linked-accounts-242401',
    'https://www.cnet.com/news/2016-fake-news-stories-truth-fiction-hoaxes/',
    'https://www.nbcnews.com/news/world/fake-news-how-partying-macedonian-teen-earns-thousands-publishing-lies-n692451',
    'https://www.news5cleveland.com/news/national/fake-news-about-the-2016-election-nets-23-year-old-a-big-payday',
    'https://www.theverge.com/2016/12/6/13850230/fake-news-sites-google-search-facebook-instant-articles',
    'http://www.telegraph.co.uk/technology/0/fake-news-exactly-has-really-had-influence/',
    'http://www.pressdemocrat.com/opinion/7425071-181/gullixson-what-happened-in-2016',
    'https://www.usatoday.com/story/tech/news/2016/11/17/report-fake-election-news-performed-better-than-real-news-facebook/94028370/',
    'https://www.vanityfair.com/news/2016/11/fake-news-russia-donald-trump',
    'http://observer.com/2017/08/facebook-fake-news/',
    'https://en.wikipedia.org/wiki/Fake_news',
    'https://www.engadget.com/2017/09/06/facebook-russian-group-spent-100-000-on-fake-news-ads/'
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

		yield ElectionItem(title=title, base_url=base, url=url, text=text, image_hash_ids=image_hashes, image_urls=image_urls)

		# # extract the 'Next' link from the pagination, load it, and
		# # parse it
		# next = response.css("div.pages").xpath("a[contains(., 'Next')]")
		# yield scrapy.Request(next.xpath("@href").extract_first(), self.parse_page_images)

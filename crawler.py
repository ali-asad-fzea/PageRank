#!/usr/bin/env python

import queue
import multiprocessing
import networkx as nx
import socket
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



def draw_graph(graph,graph_file_name):
	nx.draw(graph,with_labels=False)
	nx.drawing.nx_agraph.write_dot(graph, graph_file_name+'.dot')
	plt.savefig('./docs/'+graph_file_name+'.png')


def main(name, root_url, parser_flag = 'beautifulsoup', max_depth=1):
	def get_links_from_page(url, parser_flag):
		if parser_flag == 'beautifulsoup':
			urllist = []
			res = Request(url=url, headers={'User-Agent': 'XYZ/3.0'})
			webpage = urlopen(res, timeout=20).read()
			htmlpage=webpage
			page=BeautifulSoup(htmlpage, features="html.parser")
			for a in page.findAll("a"):
				try:
					link = a['href']
					if link[:4] in ['http', 'https']:
						urllist.append(link)
				except:
					pass
		else: print("Do not know how to parse the html !!! Specify a parser_flag")
		return urllist
	
	next_url=queue.Queue()
	crawled_urls=[]
	ip_list=[]
	g=nx.Graph()

	next_url.put((root_url,0))
	g.add_node(root_url)

	while not next_url.empty():
		url, depth = next_url.get()
		if depth >= max_depth : continue
		try:
			links = get_links_from_page(url, parser_flag)
		except Exception as e:
			links = []
			print(f"Error : {e}, URL : {link}")

		for link in links:
			g.add_node(link)
			g.add_edge(url,link)
			if link not in crawled_urls:
				next_url.put((link,depth+1))
				crawled_urls.append(link)

	for url in crawled_urls:
		try:
			ip_list.append(socket.gethostbyname(urlparse(url).netloc))
			ip_list=list(set(ip_list))
		except Exception as e:
			print(f"error : {e}   : url = {url}")
	
	stoi = {item:i for i, item in enumerate(g.nodes, 1)}
	DG = nx.DiGraph()
	for edge in g.edges():
		u = stoi[edge[0]]
		v = stoi[edge[1]]
		DG.add_edge(u, v)
	nx.write_graphml(g, f"{name}-graph-str-{depth}.graphml")
	nx.write_graphml(DG, f"{name}-graph-int-{depth}.graphml")
	return g


if __name__=='__main__':
	websites = {
		"iisc" : {"url" : "https://iisc.ac.in/", "description" : "IISc data"},
		"Wikipedia": {"url": "https://www.wikipedia.org/", "description": "Articles and links for text processing and navigation-based models."},
		"Reddit": {"url": "https://www.reddit.com/", "description": "User comments and posts for sentiment analysis and interaction patterns."},
		# "Hacker News": {"url": "https://news.ycombinator.com/", "description": "Articles, comments, and user engagement data for interaction patterns."},
		"IMDB": {"url": "https://www.imdb.com/", "description": "Movie reviews, ratings, and actor connections for network modeling."},
		# "Tripadvisor": {"url": "https://www.tripadvisor.com/", "description": "Reviews of hotels, restaurants, and attractions for sentiment analysis."},
		# "Goodreads": {"url": "https://www.goodreads.com/", "description": "Book reviews, author info, and ratings for recommendation systems."},
		# "Craigslist": {"url": "https://www.craigslist.org/", "description": "Classified listings in jobs, housing, services for user behavior modeling."},
		# "YouTube": {"url": "https://www.youtube.com/", "description": "Video metadata, comments, and user interaction for video recommendations."},
		# "Foursquare": {"url": "https://foursquare.com/", "description": "Location data and user check-ins for location-based mobility patterns."},
		# "Stack Overflow": {"url": "https://stackoverflow.com/", "description": "Question and answer data for text and interaction modeling."},
		# "Github": {"url": "https://github.com/", "description": "Repository data, commit histories, and developer activity for code evolution."},
		# "NY Times": {"url": "https://www.nytimes.com/", "description": "News articles and user comments for text-based news coverage modeling."},
		# "Quora": {"url": "https://www.quora.com/", "description": "Questions, answers, and user interactions for user behavior modeling."},
		# "BBC News": {"url": "https://www.bbc.com/news", "description": "News articles and comments for text-based topic analysis and user reactions."},
		# "Etsy": {"url": "https://www.etsy.com/", "description": "Product listings, user reviews, and ratings for shopping behavior modeling."},
		# "The Guardian": {"url": "https://www.theguardian.com/", "description": "Articles and opinion pieces for text-based analysis and topic shifts."},
		# "The Verge": {"url": "https://www.theverge.com/", "description": "Tech-related news articles and reviews for content analysis."},
		# "TechCrunch": {"url": "https://techcrunch.com/", "description": "Technology news articles and startup information."},
		# "Product Hunt": {"url": "https://www.producthunt.com/", "description": "Latest product launches and user reviews."}
	}


	parser_flag = 'beautifulsoup'
	max_depth=4
	processes = []
	for name, data in websites.items():
		process = multiprocessing.Process(target=main, args=(name, data["url"], parser_flag, max_depth))
		processes.append(process)
		process.start()

	for process in processes:
		process.join()
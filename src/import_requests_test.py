import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin

 
scraper = cloudscraper.create_scraper()
#url= "https://www.metacritic.com/game/elden-ring/"
url = "https://www.ign.com/games/elden-ring"

soup = BeautifulSoup(scraper.get(url).text, 'html.parser')
#print ("Result:", scraper.get(url).text)
#print(soup.text)


urls = []
for link in soup.find_all('a'):
    #print(link.get('href'))
    print (urljoin(url, link.get('href')))
    #print(urljoin(url, link.get['href']))
          

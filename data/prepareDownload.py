from bs4 import BeautifulSoup
import re
import urllib.request as urllib2


basePath = "https://m-selig.ae.illinois.edu/ads/"

html_page = urllib2.urlopen(f"{basePath}coord_database.html")
soup = BeautifulSoup(html_page, 'lxml')

pattern = re.compile('\.dat', re.IGNORECASE)

ind = 1
links = []

for link in soup.find_all("a", attrs={'href': pattern}):
    links.append(link.get('href'))

    urllib2.urlretrieve(basePath+link.get('href'), "./data/raw/"+link.get('href').rsplit('/',1)[-1])

print(link)



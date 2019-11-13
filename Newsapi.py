import requests
import pandas as pd
from datetime import date

today = date.today().strftime("%Y-%m-%d")
#-1 Strong Left, -0.5 Lean Left, 0 Center, 0.5 Lean Right, 1 Strong Right
NEWS_rating = {
    'abc-news': -0.5,
    'associated-press': 0,
    'breitbart-news': 1,
    'buzzfeed': -1,
    'cbs-news': -0.5,
    'cnn': -0.5,
    'fox-news': 1,
    'mashable': -1,
    'msnbc': -1 ,
    'national-review': 1,
    'new-york-magazine': -1,
    'reuters': 0,
    'the-hill': 0,
    'the-huffington-post': -1,
    'the-new-york-times': -0.5,
    'the-verge': -0.5 ,
    'the-wall-street-journal': 0.5,
    'the-washington-post': -0.5,
    'the-washington-times': 0.5,
    'time': -0.5,
    'usa-today': 0
}
def create_url(source, date):
    return ('https://newsapi.org/v2/everything?'
       'sources=' + source + '&'
        'from=' + date + '&'
        'to=' + date + '&'
        'sortBy=popularity&'
       'apiKey=36382dd42466474f9a2c3a435c63e9bf')

def capture_articles():
    All_articles = []
    for source in NEWS_rating.keys():
        url = create_url(source, today)
        try:
            articles = requests.get(url).json()["articles"]

            for a in articles:
                a['source'] = a['source']['id']
                a['bias'] = NEWS_rating[a['source']]
                if None not in a.values():
                    All_articles.append(a)
                else:
                    pass
                    #print("Contains a None", a)
        except Exception as e:
            print("some error", e)
    data_of_the_day = pd.DataFrame(All_articles)
    data_of_the_day.to_csv("data/Data"+today+".csv", index = None, header=True)
    print(data_of_the_day.shape)
"""url1 = ('https://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=36382dd42466474f9a2c3a435c63e9bf')
url2 = ('https://newsapi.org/v2/top-headlines?'
       'sources=bbc-news&'
        'from=2019-11-10&'
       'apiKey=36382dd42466474f9a2c3a435c63e9bf')

url3 = ('https://newsapi.org/v2/everything?'
       'sources=bbc-news&'
        'from=2019-11-11&'
        'sortBy=popularity&'
       'apiKey=36382dd42466474f9a2c3a435c63e9bf')


url4 = ('https://newsapi.org/v2/sources?'
       'country=us&'
       'apiKey=36382dd42466474f9a2c3a435c63e9bf')

response = requests.get(url3)
print(response.json())
try:
    articles = response.json()["articles"]
    print("Number:", len(articles))
    print(articles[0].keys())
    print(articles[0])
except:
    print("No Articles")

for a in articles:
    a['source'] = a['source']['id']
    a['bias'] = NEWS_rating[a['source']]
df = pd.DataFrame(articles)


for col in df.columns:
    print(col)

null_data = df[df.isnull().any(axis=1)]
print(null_data)
print(articles[3])
try:
    sources = response.json()["sources"]
    print("Number of sources", len(sources))
    print([x["id"] for x in sources])
except:
    print("No Sources")
"""
capture_articles()

for i in range(22, 0, -1):
    today = "2019-10-"+str(i)
    capture_articles()

import pandas as pd
import requests
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import re
import pymorphy2
from bs4 import BeautifulSoup, SoupStrainer
import plotly.express as px
from alive_progress import alive_bar

response = requests.get('http://Duma.gov.ru/news/48953')
data = response.text

soup = BeautifulSoup(response.content, 'html.parser', parse_only=SoupStrainer('div', class_='article__content'))
print(soup)
text = soup.get_text()
print(text)
print(len(text))

clear_text = re.sub(r'[^а-я\s\-]', '', text.lower())
clear_text = re.sub(r'\s', ' ', clear_text)
clear_text = re.sub(' +', ' ', clear_text)

morph = pymorphy2.MorphAnalyzer(lang='ru')
words_norm = []
words = clear_text.split()
bar_max = len(words)

with alive_bar(bar_max, force_tty=True, length=30) as bar:
    for word in words:
        p = morph.parse(word)[0]
        words_norm.append(p.normal_form)
        bar()

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
words_no_stops = [word for word in words_norm if not word in stop_words]

freq_dist = FreqDist(words_no_stops)
freq_dist_df = pd.DataFrame.from_dict(freq_dist, orient='index')

freq_dist_df.sort_values(by=[0], ascending=False, inplace=True)
freq_dist_df.reset_index(inplace=True)
freq_dist_df.columns = ['Слово', 'Частота']

print('10 самых частых слов')
freqwords = freq_dist_df.head(10)
print(freqwords)

fig = px.line(freq_dist_df, title="Закон Ципфа", x=freq_dist_df.index, y="Частота", width=1400, height=788)
fig.update_xaxes(title_text='Ранг')
fig.update_yaxes(title_text='Частота')
fig.show()

min_rang = 9
max_rang = 19
print("Ключевые слова")
keywords = freq_dist_df.iloc[min_rang:max_rang]
print(keywords)
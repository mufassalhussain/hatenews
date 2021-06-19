from flask import Flask, render_template, request
import pickle
import requests
import re
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup

app = Flask(__name__)
new_model = load_model('my_model.h5')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('home.html')

def scraper(query):

        root = "https://www.google.com/"
        url = "https://www.google.com/search?q=" + query + "&sxsrf=ALeKk029OL36aFUqo3s52mGqoyVmL_WiiA:1618659740388&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjW49m2mYXwAhWUnVwKHbUrAxEQ_AUoAnoECAIQBA&cshid=1618659749223277&biw=1242&bih=568"
        page = requests.get(url)
        page.content

        soup = BeautifulSoup(page.content, 'html.parser')

        allnews = soup.findAll(attrs={'class': 'ZINbbc xpd O9g5cc uUPGi'})

        rawlink = soup.select('.kCrYT a')
        for link in rawlink[:1]:
            raw_link = link.get('href')
            actual_link = (raw_link.split("/url?q=")[1]).split('&sa=U&')[0]
            #print(actual_link)

        title = soup.find(attrs={'class': 'BNeawe vvjwJb AP7Wnd'}).text

        description = soup.find(attrs={'class': 'BNeawe s3v9rd AP7Wnd'}).text

        time = description.split(' · ')[0]
        descript = description.split(' · ')[1]

        new_data = [title, descript, time, actual_link]
        return (new_data)

def preprocess_text(sen):

        # lower the character
        sentence = sen.lower()

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sen)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        stops = stopwords.words('english')

        for word in sentence.split():
            if word in stops:
                sentence = sentence.replace(word, '')
        return sentence


@app.route('/', methods=['POST'])
def search():

    query = request.form['query']
    #print(query)
    res = scraper(query)
    #print(res)
    actual_link = res[3]
    time = res[2]
    description = res[1]
    title = res[0]
    preprocessed_text = preprocess_text(description)
    #print(preprocessed_text)
    data = [[description, preprocessed_text]]
    df = pd.DataFrame(data, columns=['Scraped_Data', 'Preprocessed_Data'])
    #print(df)
    tw = tokenizer.texts_to_sequences(preprocessed_text)
    tw = pad_sequences(tw, maxlen=2000)
    score = new_model.predict(tw)
    hate_count = 0
    not_hate_count = 0
    for l in score:
        for item in l:
            if item < 0.1:
                #print("Not Hate")
                not_hate_count = not_hate_count + 1
            else:
                #print("Hate")
                hate_count = hate_count + 1
    total = hate_count + not_hate_count
    not_hate_percentage = (not_hate_count / total) * 100
    hate_percentage = (hate_count / total) * 100
    percentage_list = [not_hate_percentage, hate_percentage]
    return render_template('search.html', query=query, not_hate_percentage=not_hate_percentage,
                           hate_percentage=hate_percentage, percentage_list=percentage_list,
                           title=title, description=description, time=time, actual_link=actual_link)


if __name__ == '__main__':
    app.run(debug=True)
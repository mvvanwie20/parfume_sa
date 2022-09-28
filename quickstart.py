import streamlit as st
import pandas as pd
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
# -----------------------------------------------------------

# Helper functions
@st.cache
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv"
    )
    return df

df = load_data()

def sentiment(df):
    df=df[df["Review"].str.contains("Translated by Google")==False]
    df=df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    review_list=df['Review'].values.tolist()
    
    lines_list = tokenize.sent_tokenize(str(review_list))
    
    sia = SentimentIntensityAnalyzer()
    sia_scores=sia.polarity_scores(str(lines_list))
    del sia_scores['compound']
    
    labels = []
    sizes = []

    for x, y in sia_scores.items():
        labels.append(x)
        sizes.append(y)

    plt.style.use('fivethirtyeight')

    plt.pie(sizes, labels=labels,)
    plt.title('Sentiment of Reviews')
    plt.axis('equal')
    plt.show()


def common_words(df):
    df=df[df["Review"].str.contains("Translated by Google")==False]
    df=df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    review_list=df['Review'].values.tolist()
    
    tokenizer = RegexpTokenizer(r'\w+')
    token_list=tokenizer.tokenize(str(review_list))
    
    list = [''.join(x for x in i if x.isalpha()) for i in token_list] 
    
    tags = nltk.pos_tag(no_num)
    descriptors = [word for word,pos in tags if (pos == 'JJ'or pos == 'NN')]
    
    counterd=Counter(descriptors)
    
    top20words= counterd.most_common(20)
    
    plt.style.use('fivethirtyeight')
    plt.xlabel("Count")
    plt.ylabel("Words")  
    plt.title("Most Popular(helpful) Words")
    plt.barh(words, counts)
    
    plt.show()

# Main

st.title("A Look at Parfumado's Reviews")

st.write("Here is the dataset used in this analysis:")

df_display = st.checkbox("Display Raw Data", value=True)

if df_display:
    st.write(df)

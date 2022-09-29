import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import punkt
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt



# Helper functions
# -----------------------------------------------------------
df1 = pd.read_csv('reviewscsv.csv')


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

    plt.style.use('ggplot')

    plt.pie(sizes, labels=labels,autopct='%1.1f%%')
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
    
    tags = nltk.pos_tag(list)
    descriptors = [word for word,pos in tags if (pos == 'JJ'or pos == 'NN')]
    
    counterd=Counter(descriptors)
    
    top20words= counterd.most_common(20)
    words = []
    counts = []
    for item in top20words:
         words.append(item[0])
         counts.append(item[1])
    plt.style.use('ggplot')
    plt.xlabel("Count")
    plt.ylabel("Words")  
    plt.title("Most Popular(helpful) Words")
    plt.barh(words, counts)

    plt.show()

# Main
#---------------------------------------------------

st.title("A Look at Parfumado's Reviews")

st.write("Here is the dataset used in this analysis:")

df_display = st.checkbox("Display Raw Data", value=True)


if df_display:
    st.write(df1)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Sentiment Analysis')
st.write("Below are the results of the senitment analysis of the reviews")
st.pyplot(sentiment(df1))

st.header('Top 20 Words')
st.write("Below is a visualization of the 20 most used descriptive and object words and their count")

st.pyplot(common_words(df1))


import streamlit as st
import pandas as pd
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import imageio as iio



# Helper functions
# -----------------------------------------------------------
#Upload the csv file from github
df1 = pd.read_csv('reviewscsv.csv')

#Upload image
png=iio.imread("word-cloud.png")

#function that graphs reviews by month
def months(df):
    #convert df column to list and then convert to datetime and simplify to just year and date
    dates=df['Date'].values.tolist()
    idx = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
    
    #create df of year/months and their counts of occurence
    df2 = idx.value_counts().sort_index().reset_index()
    df2.columns = ['Date','Count']
    
    #set x and y values
    x=df2['Date']
    y=df2['Count']
    
    #plot graph
    plt.style.use('ggplot')
    plt.xlabel("Months")
    plt.ylabel("Number of Reviews")  
    plt.title("Reviews by Month since 2017")
    plt.xticks(rotation=90)
    plt.plot(x,y)
    plt.show()
    
def sentiment(df):
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    
    #clean data of translations and emojis
    df=df[df["Review"].str.contains("Translated by Google")==False]
    df=df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    #convert reviews to list and tokenize into sentences
    review_list=df['Review'].values.tolist()
    
    lines_list = tokenize.sent_tokenize(str(review_list))
    
    #use vader sentiment analysis on tokens, delete compound score bc its not necessary for pie chart
    sia = SentimentIntensityAnalyzer()
    sia_scores=sia.polarity_scores(str(lines_list))
    del sia_scores['compound']
    
    #add labels and move vlaues to list
    labels = ['Negative','Neutral','Positive']
    sizes = []

    for x, y in sia_scores.items():
        sizes.append(y)
        
    #plot graph
    plt.style.use('ggplot')
    explode= [0, 0.1, 0]
    plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%',shadow=True)
    plt.title('Sentiment of Reviews')
    plt.axis('equal')
    plt.show()


def common_words(df):
    import nltk
    nltk.download('averaged_perceptron_tagger')
    
    #clean data of emojis and translated texts
    df=df[df["Review"].str.contains("Translated by Google")==False]
    df=df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    #convert to list and tokenize by word
    review_list=df['Review'].values.tolist()
    
    tokenizer = RegexpTokenizer(r'\w+')
    token_list=tokenizer.tokenize(str(review_list))
    
    #Keep only words remove numbers
    list = [''.join(x for x in i if x.isalpha()) for i in token_list] 
    
    #create part of speach tags and keep only adj and nouns
    tags = nltk.pos_tag(list)
    descriptors = [word for word,pos in tags if (pos == 'JJ'or pos == 'NN')]
    
    #count occurences of words
    counterd=Counter(descriptors)
    
    #take top 20 (in this case 21 and remove the falsely classifed value)
    top20words= counterd.most_common(21)
    del top20words[8]
    
    #Make words and counts list for graph
    words = []
    counts = []
    for item in top20words:
         words.append(item[0])
         counts.append(item[1])
    
    #plot graph
    plt.style.use('ggplot')
    plt.xlabel("Count")
    plt.ylabel("Words")  
    plt.title("Most Popular(filtered) Words")
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

st.header('Reviews Over Time')
st.write("""Below are the reviews organized by time. The data was split into the months the reviews were
written and given counts of occurences. This data was then plotted as seen below. March of 2021 had the highest spike 
reviews posted, perhaps this correlated with a campaign.""")
st.pyplot(months(df1))

st.header('Sentiment Analysis')
st.write("""Below are the results of the sentiment analysis of the reviews. 
Reviews were broken up into sentences and then analyzed using NLTK's VADER sentiment analysis module
to calculate their general tone. The majority of messages have a neutral characteristics, but positive sentiments 
significantly outweigh the negative in this investigation.""")
st.pyplot(sentiment(df1))

st.header('Top 20 Words')
st.write("""Below is a visualization of the 20 most used descriptive and object words and their count. 
         This visualization was based on these categories as a way to remove frequent common words, 
         such as conjunctions and pronouns, and helps us identify common perceptions of Parfumado.""")

st.pyplot(common_words(df1))

st.header('Most Used Words')
st.write("""Finally, we have a word cloud of the most frequent words over all; Just as you mentioned Martijn :)""")

st.image(png)


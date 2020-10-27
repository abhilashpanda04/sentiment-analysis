import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
import spacy
import string
import nest_asyncio
import twint
import re
import nltk
################################################

#Getting the tweets without any API
nest_asyncio.apply()
c = twint.Config()
c.Search = "Work from home"
c.Store_object = True
#c.Since="2020-09-01"
c.Store_csv = True
c.Store_pandas = True
c.Pandas = True
c.Output = "tweets.csv"
c.lang= "en"
c.Limit = 1000
some=twint.run.Search(c)
tlist = c.search_tweet_list
print("hacked that out")


###################################################
#Getting the tweets into a dataframe as it is in a dictionary list

df = pd.DataFrame(tlist)


#column that has all the tweets

df["tweet"]

df.to_csv("tweet sentiment.csv")

#keeping the dataframe short and simple with  only tweets and username

tweet=pd.DataFrame(df[['tweet','username']])
tweet

#Not loosing out the origional dataframe and maing a duolicate

retweet=tweet

#Started importing all the NLTK modules

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('stopwords')
#Making a Lemmetiser and a stemmer object to use at a later part

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords=stopwords.words('english')
nltk.download('wordnet')
#cleaning the text with a function

def preprocess(sentence):
    sentence=str(sentence)#making it string
    sentence = sentence.lower()#making it lowercase
    sentence=sentence.replace('{html}',"") #removing any word with HTML
    cleanr = re.compile('<.*?>')#putting regular expression if . is there remove it
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)#removing HTTPS
    rem_num = re.sub('[0-9]+', '', rem_url)#remove all the numbers
    tokenizer = RegexpTokenizer(r'\w+')#using regex tokenizer object
    tokens = tokenizer.tokenize(rem_num)#removing number
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords]#keeo the words if word is greater than 2 letters and not in stopwords
    stem_words=[stemmer.stem(w) for w in filtered_words]#stemming the words
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]#putting Lemmetiser into words to know the root stem_words
    return " ".join(filtered_words)#putting into a different column


retweet['clean']=retweet['tweet'].apply(lambda x:preprocess(x)) #so calling the avbove cleaning function to clean text and keeping it at a new column

retweet['clean']#checking how the clean text looks like


retweet.to_csv('retweet.csv')#keeping the data frame with a csv to read all columns as string

tweet = pd.read_csv(r'retweet.csv')#reading it as csv again
tweet#checking the dataframe


analyzer = SentimentIntensityAnalyzer()#starting to build an analyzer object to measure sentiment

sentiment= tweet['clean'].apply(lambda x: analyzer.polarity_scores(x))#started analysing the clen tweets and assigning score


sentiment#checing how the sentiment score looks like :) :)




retweet = pd.concat([retweet, sentiment.apply(pd.Series)],1) #putting it into a lambda function and appending into the existing data frame

retweet #checking the appended dataframe

retweet.drop_duplicates(subset = 'twxt',inplace = True)


#visualisation with pandas
retweet['compound'].mean() #checking mean
retweet['compound'].hist()# checking histogram
retweet['compound'].median()#checking median

#checking with seaborn

ax1 = sns.distplot(retweet['compound'], bins=15, hist = False, label = 'work rom home', color = 'r', kde_kws={'linestyle':'--'})

plt.legend()
plt.title('work rom home')

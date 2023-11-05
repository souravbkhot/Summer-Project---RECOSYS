# # Libraries
# import pandas as pd
# import numpy as np
# import warnings 
# from pprint import pprint
# import warnings
# warnings.filterwarnings("ignore")
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stopwords = stopwords.words('english')
# # nltk.download('omw-1.4')
# #for data wrangling and manipulation

#for data wrangling and manipulation

#for data wrangling and manipulation

import pandas as pd
import numpy as np

#for NLP text processing and formatting

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# For word lemmitization
import nltk
# for word Stemming
nltk.download('omw-1.4')


from textblob import TextBlob
from nltk.tokenize import word_tokenize

# Global Parameters
# from nltk.corpus import wordnet as wn
# wn.ensure_loaded()
stop_words = set(stopwords.words('english'))


import warnings
warnings.filterwarnings("ignore")

from geopy.geocoders import Nominatim



import streamlit as st
import pickle

Cuisine_list= pickle.load(open('my_dict.pkl','rb'))
st.title("RECOSYS")

#  Data
data=pd.read_csv("C:\Dataset\Data/RECOSYS_reviews.csv")

se = data['categories/0'].unique()
ser0 = set(se)
s= data['categories/0'].nunique()

se = data['categories/1'].unique()
ser1 = set(se)
s=data['categories/1'].nunique()

se = data['categories/2'].unique()
ser2 = set(se)
s=data['categories/2'].nunique()

se = data['categories/3'].unique()
ser3= set(se)
s=data['categories/3'].nunique()

se = data['categories/4'].unique()
ser4= set(se)
s=data['categories/4'].nunique()

se = data['categories/5'].unique()
ser5= set(se)
s=data['categories/5'].nunique()

se = data['categories/6'].unique()
ser6= set(se)
s=data['categories/6'].nunique()

se = data['categories/7'].unique()
ser7= set(se)
s=data['categories/7'].nunique()

se = data['categories/8'].unique()
ser8= set(se)
s=data['categories/8'].nunique()

se = data['categories/9'].unique()
ser9= set(se)
s=data['categories/9'].nunique()

# For all unique Cuisines 
ser=ser0|ser1|ser2|ser3|ser4|ser5|ser6|ser7|ser8|ser9 

Cuisine=set(filter(lambda x: x == x , ser))   #For removing nan value from set

x = st.selectbox(
"Select the Cuisine: ",
Cuisine_list)

user_input = st.text_input("Enter Your Location:", "Pune University, Pune")
if st.button("Search"):

    Sel_Cuisine=data.loc[(data['categories/0']== x)| (data['categories/1']== x)|(data['categories/2']== x)|(data['categories/3']== x)|(data['categories/4']== x)|(data['categories/5']== x)|(data['categories/6']== x)|(data['categories/7']== x)|(data['categories/8']==x)|(data['categories/9']== x)]


    sentiment=Sel_Cuisine.iloc[:,[11,12,13,16,18,19]]


    sentiment['text']= sentiment['text'].astype(str)   #for converting string

    sentiment.replace('nan', np.nan, inplace = True)
    sentiment= sentiment.dropna()

    def is_special(text):                # Remove special characters¶
        rem = ''
        for i in text:
            if i.isalnum():
                rem = rem + i
            else:
                rem = rem + ' '
        return rem
    sentiment.text = sentiment.text.apply(is_special)

    def to_lower(text):    # Convert everything to lowercase
        return text.lower()

    sentiment.text = sentiment.text.apply(to_lower)

    sentiment.text=sentiment.text.apply(str)
    def rem_stopwords(text):                                   # remove stopwords 
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        return [w for w in words if w not in stop_words]
    sentiment.text = sentiment.text.apply(rem_stopwords)

    def stem_txt(text):                                       #stem words
        ss = SnowballStemmer('english')
        return " ".join([ss.stem(w) for w in text])

    sentiment.text = sentiment.text.apply(stem_txt)

    def calcPolarity(text):                              #Polarity
        return TextBlob(text).sentiment.polarity

    def calcSubjectivity(text):                          #Subjectivity
        return TextBlob(text).sentiment.subjectivity

    def segmentation(text):
        if text > 0:
            return 'positive'
        elif text == 0 :
            return 'neutral'
        else:
            return 'negative'
        
    sentiment['tPolarity']=sentiment['text'].apply(calcPolarity)
    sentiment['tSubjectivity']=sentiment['text'].apply(calcSubjectivity)
    sentiment['segmentation']=sentiment['tPolarity'].apply(segmentation)

    def sRate(tPolarity):       #Formula for calculating sentiment based star rating
        rating=(((tPolarity - (-1))*(5-0))/(1-(-1)))+0
        return rating

    sentiment['Srating']=sentiment['tPolarity'].apply(sRate)
    final=sentiment.groupby(['location/lat','location/lng','title','address'],as_index=False).agg({'text':'count','Srating':'mean'}).sort_values(by="Srating",ascending=False)
    Restro=final[final['text'] >=100]
    
    # calling the Nominatim tool
    loc = Nominatim(user_agent="GetLoc")
    
    # entering the location name
    getLoc = loc.geocode(user_input)

    from math import sin, cos, sqrt, atan2, radians

    # Approximate radius of earth in km
    R = 6373.0

    def resto_loc(res_lat,res_long):
        lat1 = radians(getLoc.latitude)
        lon1 = radians(getLoc.longitude)
        lat2 = radians(res_lat)
        lon2 = radians(res_long)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        return distance
    Restro['distance']=Restro.apply(lambda x: resto_loc(x['location/lat'], x['location/lng']),axis=1)
    RECOSYS=Restro[Restro['distance'] <=5]
    RECOSYS=RECOSYS.iloc[:,[2,3,5,6]]
    RECOSYS=RECOSYS.head(10)
    print(RECOSYS)
    st.dataframe(RECOSYS)

agri="""
<style>
[data-testid="stAppViewContainer"]{
    background-image:url("https://img.freepik.com/fotos-premium/fundo-de-alimentos-com-lugar-para-texto-com-diferentes-tipos-de-massas-tomates-ervas-cogumelos-ovos-temperos-espalhados-na-luz-de-fundo-de-marmore-vista-superior-conceito-de-cozinha-italiana_90258-3528.jpg?w=2000");
    background-size: cover;
}
</style>
"""
st.markdown(agri,unsafe_allow_html=True)



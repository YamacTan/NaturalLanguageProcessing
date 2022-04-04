#######################
# Yamac TAN - Data Science Bootcamp - Week 12 - pROJECT1
#######################

# %%

import numpy as np
import pandas as pd
from PIL import Image
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# %%
###############################################
# Görev 1
###############################################

df = pd.read_csv("Odevler/HAFTA_12_NLP/wiki_data.csv")
df.head(10)
df.columns
df['text']  # Column that where our texts are located.

# Görev 1 - Adım 1:

def clean_text(df, textColumnName):
    df[textColumnName] = df[textColumnName].str.lower()
    df[textColumnName] = df[textColumnName].str.replace('[^\w\s]', '')  #[^\w\s] is the regex expression for punctuations
    df[textColumnName] = df[textColumnName].str.replace('\d', '')


    return df

# Görev 1 - Adım 2:
cleaned_df = clean_text(df,'text')

# Görev 1 - Adım 3:
stopwords = stopwords.words('english')

def remove_stopwords(df,textColumnName,stopwords):
    df[textColumnName] = df[textColumnName].apply(lambda x: " ".join(x for x in str(x).split() if x not in stopwords))
    return df

# Görev 1 - Adım 4:
cleaned_df = remove_stopwords(cleaned_df,'text',stopwords)

# Görev 1 - Adım 5:

word_counts = pd.Series(' '.join(cleaned_df['text']).split()).value_counts()

rare_words = word_counts[word_counts <= 1500]
cleaned_df['text'] = cleaned_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))

# Görev 1 - Adım 6:

cleaned_df["text"].apply(lambda x: TextBlob(x).words).head()
# As a result, we can see that each sentence is divided into tokens.

# Görev 1 - Adım 7:

cleaned_df['text'] = cleaned_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# %%
###############################################
# Görev 2
###############################################

# Görev 2 - Adım 1

frequency = cleaned_df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
frequency.columns = ["words", "frequency"]
frequency.sort_values("frequency", ascending=False)


# Görev 2 - Adım 2

frequency[frequency["frequency"] > 7500].plot.bar(x="words", y="frequency")
plt.show()

# Görev 2 - Adım 3

text = " ".join(i for i in cleaned_df.text)

wordcloud = WordCloud(max_font_size=50,
                      max_words=200,
                      background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
###############################################
# Görev 3
###############################################


def nlp_preparation(df,textColumnName, stopwords, boxplot = False, wordcloud = False):

    """

    This function performs the necessary data preprocessing steps and data visualizations on pandas dataframes that
    are planned to be used for natural language processing. Pandas dataframe name, column name with texts in dataframe
    and stopwords list to be used are entered as parameters. Options for visualization steps can be activated with
    boolean parameters.


    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        Dataframe to work on.

    textColumnName: str
        Column name with texts in the dataframe.

    stopwords: nltk.corpus.reader.wordlist.WordListCorpusReader
        List of stopwords to be used. Lists prepared in different languages and with different modifications can be used.

    boxplot: bool, optional, default: False
        Boolean parameter to use for displaying boxplot output.

    wordcloud: bool, optional, default: False
        Boolean parameter to use for displaying wordcloud output.

    Examples
    -------
        import pandas as pd
        import nltk
        import matplotlib.pyplot as plt
        from textblob import Word, TextBlob
        from wordcloud import WordCloud
        from nltk.corpus import stopwords
        df = pd.read_csv("Odevler/HAFTA_12_NLP/wiki_data.csv")
        stop_words = stopwords.words('english')
        nlp_preparation(df,'text',stop_words, boxplot = True, wordcloud = True)


    Returns
    -------
    None

    """

    # Date preprocessing
    clean_text(df,textColumnName)
    remove_stopwords(df, textColumnName, stopwords)

    word_counts = pd.Series(' '.join(df[textColumnName]).split()).value_counts()
    rare_words = word_counts[word_counts <= 1500]
    df[textColumnName] = df[textColumnName].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))
    df[textColumnName].apply(lambda x: TextBlob(x).words).head()
    df[textColumnName] = df[textColumnName].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    frequency = df[textColumnName].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    frequency.columns = ["words", "frequency"]
    frequency.sort_values("frequency", ascending=False)

    if boxplot == True:
        frequency[frequency["frequency"] > 7500].plot.bar(x="words", y="frequency")
        plt.show()
    if wordcloud == True:
        text = " ".join(i for i in cleaned_df.text)

        wordcloud = WordCloud(max_font_size=50,
                              max_words=200,
                              background_color="black").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()



df_new = pd.read_csv("Odevler/HAFTA_12_NLP/wiki_data.csv")
stop_words = stopwords.words('english')
nlp_preparation(df_new,'text',stop_words, boxplot = True, wordcloud = True)



























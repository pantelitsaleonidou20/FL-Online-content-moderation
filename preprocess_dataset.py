
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import re
import string
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
# TEXT_CLEANING function: conducts text cleaning by:
# 1. removing tags
# 2. removing links
# 3. removing numbers
# 4. removing punctuation
# 5. removing non-ascii characters
# 6. converting to lowercase
# 7. removing white spaces other than single space
# 8. removing numbers (again)
# 9. removing English stop words
# 10. remove words that appear only once (misspelled words)

def text_cleaning(tweet_text):
    #print("preprocessing text is in process...")

    # remove tags starts with: @*
    tweet_text = ' '.join(filter(lambda x: not x.startswith('@'), tweet_text.split()))

    # remove links starts with: @http
    tweet_text = ' '.join(filter(lambda x: not x.startswith('http'), tweet_text.split()))

    # remove numbers
    tweet_text = re.sub(r'\d +', '', tweet_text)
    tweet_text = re.sub("\d+", "", tweet_text)


    # remove punctuation
    tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))

    #remove non-ascii characters
    only_ascii_tweet_text = ""

    for character in tweet_text:
        if character.isascii():
            only_ascii_tweet_text += character
    tweet_text=only_ascii_tweet_text

    # convert to lowercase
    tweet_text = tweet_text.lower()

    # remove white spaces instead of space
    tweet_text = tweet_text.replace('\t', ' ')
    tweet_text = tweet_text.replace('\n', ' ')
    tweet_text = tweet_text.replace('\v', ' ')
    tweet_text = tweet_text.replace('\f', ' ')
    tweet_text = tweet_text.replace('\r', ' ')


    # remove numbers
    tweet_text = re.sub(r'\d +', '', tweet_text)
    tweet_text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet_text)
    tweet_text = re.sub("\d+", " ", tweet_text)

    # remove stop words in english
    text_tokens = word_tokenize(tweet_text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    separator = ' '
    tweet_text = separator.join(tokens_without_sw)

    return tweet_text

def remove_words_appear_only_once(df):
    
    #join all text together and split by whitespaces
    all_val = ' '.join(df['preprocessed_text']).split()

    #get unique values
    once = [x for x in all_val if all_val.count(x) == 1]
    #print("only once:\n",str(once))

    #remove from text by nested list comprehension
    df['preprocessed_text'] = [' '.join([y for y in x.split() if y not in once]) for x in df['preprocessed_text']]

    #apply alternative
    #df['Text'] = df['Text'].apply(lambda x: ' '.join([y for y in x.split() if y not in once]))
    return df

###########################################################################################

#read dataset to preprocess

fields = ['text', 'label','rate']
df = pd.read_csv('hatespeech_text_label_vote_RESTRICTED_100K.csv', sep='\t', usecols=fields)

#remove records with missing values
df=df.dropna()

#remove columns we don't need
df=df.drop(columns=['rate'])

print(len(df))

#print("spam len:",len(df[df['label']=="spam"]))
print("abusive:",len(df[df['label']=="abusive"]))
print("spam:",len(df[df['label']=="spam"]))
print("hateful", len(df[df['label']=="hateful"]))
print("normal len:",len(df[df['label']=="normal"]))


#remove spam tweets from dataset
df = df[df['label'] != 'spam']

#convert two binary classes 0-normal 1-inappropriate
df.loc[df['label'] == 'normal', 'label'] = 0
df.loc[df['label'] == 'abusive', 'label'] = 1
df.loc[df['label'] == 'hateful', 'label'] = 1

    

print("inappropriate: ", len(df[df['label'] == 1]) )
print("normal: ", len(df[df['label'] == 0]) )


df_clean=pd.DataFrame(columns=['text','label'])
count=0
for index, row in df.iterrows():
    row=pd.DataFrame([[row['text'],text_cleaning(row['text']),row['label']]],columns=["text","preprocessed_text","label"])
    df_clean=pd.concat([df_clean, row])
    count=count+1
    if count % 100 ==0:
        print(count)

df=df_clean.dropna()
print(len(df))

#call function to remove words that only appears once
df= remove_words_appear_only_once(df)

#remove records with missing columns
df=df.dropna()

#remove empty strings/texts
df = df[df['preprocessed_text'] != '']
df = df.replace('nan', np.nan)
df = df.dropna()

print("inappropriate:",len(df[df['label']==1]))
print("normal:",len(df[df['label']==0]))
print(len(df))


df.to_csv("abusive_preprocessed_all.csv", index=False)

exit(0)

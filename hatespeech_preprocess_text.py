import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re
import string

# text preprocessing
def text_preprocessing(tweet_text):
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
    # print("no punctuation: ",tweet_text)

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


import pandas as pd


fields = ['text', 'label','rate']

df = pd.read_csv('hatespeech_text_label_vote_RESTRICTED_100K.csv', sep='\t', usecols=fields)

df=df.dropna()

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

    


print("inappropriate:",len(df[df['label']==1]))
print("normal:",len(df[df['label']==0]))


#output_file = open("abusive_dataset_preprocessed.csv", "a+", encoding='utf-8')
#output_file.write("text,label\n")


df_prep=pd.DataFrame(columns=['text','label'])
count=0
for index, row in df.iterrows():
    #output_file.write(str(text_preprocessing(row['text'])) + "," + str(row['label'])+ "\n")
    row=pd.DataFrame([[text_preprocessing(row['text']),row['label']]],columns=["text","label"])
    df_prep=pd.concat([df_prep, row])
    count=count+1
    if count % 100 ==0:
        print(count)
    

df=df_prep.dropna()
print(len(df_prep))

#join all text together and split by whitespaces
all_val = ' '.join(df['text']).split()
#get unique values
once = [x for x in all_val if all_val.count(x) == 1]
print("only once:\n",str(once))
#remove from text by nested list comprehension
df['text'] = [' '.join([y for y in x.split() if y not in once]) for x in df['text']]
#apply alternative
#df['Text'] = df['Text'].apply(lambda x: ' '.join([y for y in x.split() if y not in once]))

df=df.dropna()
print("this is empty: ",len(df[df['text']=='']))
df = df[df['text'] != '']


print("inappropriate:",len(df[df['label']==1]))
print("normal:",len(df[df['label']==0]))
print(len(df))


df.to_csv("abusive_preprocessed.csv",index=False)


exit()

from collections import Counter

with open("myfile.txt") as input_file:
    word_counts = Counter(word for line in input_file for word in line.split())
[word for (word, count) in word_counts.iteritems() if count==1]

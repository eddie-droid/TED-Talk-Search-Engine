import nltk
import os
import string
import re
import csv
import pickle
import numpy as np
from os.path import exists
import inquirer

CSV_PATH = "/home/evaldez/Downloads/ted_talks_en.csv"

stopwords = nltk.corpus.stopwords.words("english")


def create_dict():
    transcripts = []
    csv_dict = {}
    with open(CSV_PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # print(len(row))
            # print(row[-1])
            # print(', '.join(row))
            if row[-1] == "transcript":
                continue

            text = row[-1]
            # lower case
            text = text.lower()
            # remove punctuation
            text = "".join([char for char in text if char not in string.punctuation])
            # print(text)
            # tokenization
            words = nltk.word_tokenize(text)
            # remove stopwords
            removed = [word for word in words if word not in stopwords]
            # stemming
            porter = nltk.stem.porter.PorterStemmer()
            stemmed = [porter.stem(word) for word in removed]
            
            
            # title stemming
            title = row[1]
            # lower case
            title = title.lower()
            # remove punctuation
            title = "".join([char for char in title if char not in string.punctuation])
            # print(text)
            # tokenization
            title_words = nltk.word_tokenize(title)
            # remove stopwords
            removed = [word for word in title_words if word not in stopwords]
            # stemming
            title_stemmed = [porter.stem(word) for word in removed]
            
            descr = row[-2]
            # lower case
            descr = descr.lower()
            # remove punctuation
            descr = "".join([char for char in descr if char not in string.punctuation])
            # print(text)
            # tokenization
            descr_words = nltk.word_tokenize(descr)
            # remove stopwords
            descr_removed = [word for word in descr_words if word not in stopwords]
            # stemming
            descr_stemmed = [porter.stem(word) for word in removed]
            
            
            # [stemmed words list, document id, title]
            transcripts.append([stemmed, row[0], title_stemmed, descr_stemmed])
            # print(stemmed)
            csv_dict[row[0]] = [title_stemmed, row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[14], row[16], descr_stemmed, stemmed]
            
    with open('csv_dict.pkl', 'wb') as f:
        pickle.dump(csv_dict, f)
    
    with open('csv_dict.pkl', 'rb') as f:
        csv_dict = pickle.load(f)
    return csv_dict



def get_df(csv_dict):
    DF = {}
    for key, value in csv_dict.items():
        # tokens = transcript + description
        tokens = value[-1] + value[-2]
        for w in tokens:
            try:
                DF[w].add(key)
            except:
                DF[w] = {key}

    for i in DF:
        DF[i] = len(DF[i])

    print(DF)
    print(len(DF))

    with open('DF_dict.pkl', 'wb') as f:
        pickle.dump(DF, f)


def tf_idf(csv_dict):
    with open('DF_dict.pkl', 'rb') as f:
        DF = pickle.load(f)

    words_count = len(DF)
    tf_idf = {}
    N = 4005

    for key, value in csv_dict.items():
        tokens = value[-1]
        counter = Counter(tokens + value[0] + value[-2])
        for token in np.unique(tokens):
            tf = counter[token] / words_count  # number of times t appears in d
            df = DF[token]  # number of d's t is present in
            idf = np.log(N / (df + 1))
            
            tf_idf[key, token] = tf * idf
    
    with open('tf_idf.pkl', 'wb') as f:
        pickle.dump(tf_idf, f)
        
    return tf_idf


def Counter(tokens):
    freq_counter = {}

    for word in tokens:
        try:
            freq_counter[word] += 1
        except:
            freq_counter[word] = 1
    return freq_counter


def matching_score(query, tf_idf):
    #lowercase
    text = query.lower()

    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # tokenization
    words = nltk.word_tokenize(text)

    # remove stopwords
    removed = [word for word in words if word not in stopwords]

    # stemming
    porter = nltk.stem.porter.PorterStemmer()
    query = [porter.stem(word) for word in removed]


    query_weights = {}
    for key in tf_idf:
        if key[1] in query:
            if key[0] not in query_weights:
                query_weights[key[0]] =  tf_idf[key]

            else:
                query_weights[key[0]] += tf_idf[key]
    # print(query_weights)
    return query_weights


def toplist(query_weights, csv_dict, method):
    if method == "Relevant":
        newlist = []
        sorted_list = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_list)
        counter = int(0)
        for i in range(len(sorted_list)):
            if counter < 10:
                newlist.append(sorted_list[i][0])
                counter += 1
        # print(newlist)
        return newlist
    else:
        newlist = []
        views = {}
        for id in query_weights.keys():
            views[id] = csv_dict[id][6]
        sorted_list = sorted(views.items(), key=lambda x: int(x[1]), reverse=True)
        # print(sorted_list)
        counter = int(0)
        for i in range(len(sorted_list)):
            if counter < 10:
                newlist.append(sorted_list[i][0])
                counter += 1
        # print(newlist)
        return newlist


if __name__ == "__main__":
    query = input("\nTED Talks Search Engine\n\nEnter query: ")
    questions = [
    inquirer.List('method',
                    message="Search Method",
                    choices=['Relevant', 'Most Viewed'],
                ),
    ]
    method = inquirer.prompt(questions)["method"]
    
    print("Searching...\n")


    if not exists("csv_dict.pkl"):
        csv_dict = create_dict()
        get_df(csv_dict)
        tfidf = tf_idf(csv_dict)
    else:
        with open('csv_dict.pkl', 'rb') as f:
            csv_dict = pickle.load(f)
        with open('tf_idf.pkl', 'rb') as f:
            tfidf = pickle.load(f)

    weights = matching_score(query, tfidf)
    top_id = toplist(weights, csv_dict, method)
    print("Top Results:\n")
    for i in range(len(top_id)):
        id = top_id[i]
        tedtalk = csv_dict[id]
        print(str(i + 1) + ".", "\nTitle: " + tedtalk[1], "\nSpeaker: " + tedtalk[3], "\nViews: " + tedtalk[6], "\nURL: " + tedtalk[-3] + "\n")
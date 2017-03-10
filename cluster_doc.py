import numpy as np
#import lda
import pickle
import nltk
import math
#import lda.datasets
import string
import logging
import timeit

from TDTReader import TDTCorpus
from ChoiCorpus import ChoiCorpus
from TDTReader import Segment

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *

from gensim import corpora
import gensim

from WE import WordEmbeddings

###########
##HELPERS##
###########
def preprocess_text(text):
    stopwordlist = stopwords.words('english')

    text = text.replace('\'',' ')
    text = text.translate(None,string.punctuation)
    text = text.lower()
    text = text.split()

    filtered_text = [word for word in text if word not in stopwordlist]

    #stemmer = PorterStemmer()
    #stemmed_text = [stemmer.stem(word) for word in filtered_text]


    #return stemmed_text
    return filtered_text

def get_corpus_vocab(corpus):
    vocab = dict()
    word_count = dict()
    idx = 0
    for key, doc in corpus.iteritems():
        for sent in doc:
            for word in sent:
                if word not in vocab:
                    vocab[word] = idx
                    word_count[word] = 1
                    idx += 1
                else:
                    word_count[word] += 1
    return vocab, word_count

def clean_corpus(corpus):
    for i in range(len(corpus.dev_set)):
        key = corpus.test_set[i].strip()
        corpus.dev_set[i] = key

    num_samples = 0
    for key in corpus.text_corpus_bnds.keys():
        if key not in corpus.dev_set or num_samples >= 10:
            corpus.text_corpus.pop(key,None)
            corpus.text_corpus_bnds.pop(key,None)
            corpus.doc_boundaries.pop(key,None)
            corpus.sent_boundaries.pop(key,None)
            corpus.char_boundaries.pop(key,None)
        else:
            num_samples += 1
    return corpus

############
##FEATURES##
############
def _get_stories_list(file_names, documents, boundaries):
    stories = []
    for key in file_names:
        doc = documents[key]
        bnds = boundaries[key]

        story = []
        for sent, is_bnd in zip(doc,bnds):
            if not is_bnd:
                story = story + sent
            else:
                story = story + sent
                stories.append(story)
                story = []

    return stories

def get_corpus_formatted(corpus):
    print('loading sentences...')
    documents = dict()

    for key in corpus.text_corpus_bnds.keys():
        key = key.strip()
        val = corpus.text_corpus_bnds[key]
        document = []


        text = ' '.join(val)
        text = text.split('<bnd>')

        for sent in text:
            sent = preprocess_text(sent)
            document.append(sent)
        documents[key] = document
    corpus.text_corpus_bnds.clear()

    vocab, word_counts = get_corpus_vocab(documents)

    return documents, vocab, word_counts

#SLOW/DEPRECATED#
def get_train_test_stories(corpus):
    print('loading sentences...')
    documents = dict()

    for key in corpus.text_corpus_bnds.keys():
        key = key.strip()
        val = corpus.text_corpus_bnds[key]
        document = []


        text = ' '.join(val)
        text = text.split('<bnd>')

        for sent in text:
            sent = preprocess_text(sent)
            document.append(sent)
        documents[key] = document
    corpus.text_corpus_bnds.clear()

    vocab = get_corpus_vocab(documents)
    train_stories = _get_stories_list(corpus.train_set, documents, corpus.sent_boundaries)
    test_stories = _get_stories_list(corpus.test_set, documents, corpus.sent_boundaries)

    dev_stories = _get_stories_list(documents.keys(), documents, corpus.sent_boundaries)
    return train_stories,dev_stories, test_stories, vocab

def get_list_stories_fast(documents, boundaries, file_list, vocab):
    vocab_size = len(vocab.keys())
    stories = []

    for file_name in file_list:
        document = documents[file_name]
        boundary = boundaries[file_name]

        story = []
        for sent, is_bnd in zip(document, boundary):
            story += sent

            if is_bnd:
                stories.append(story)
                story = []


        del documents[file_name]
        del boundaries[file_name]
    return stories

def get_doc_bow_features_fast(documents, boundaries, file_list, vocab):
    vocab_size = len(vocab.keys())
    stories_bow = []

    for file_name in file_list:
        document = documents[file_name]
        boundary = boundaries[file_name]

        story_bow = np.array([1]*vocab_size)

        for sent, is_bnd in zip(document, boundary):
            idxs = [vocab[x] if x in vocab else vocab['<unk>'] for x in sent]
            story_bow[idxs] += 1

            if is_bnd:
                stories_bow.append(story_bow)
                story_bow = np.array([1]*vocab_size)
    return np.array(stories_bow)

def calculate_vocab_coverage(vocab, documents):
    total_words = 0
    covered_words = 0

    for key, doc in documents.iteritems():
        for sent in doc:
            for word in sent:
                total_words += 1
                if word in vocab:
                    covered_words += 1

    return (covered_words*1.0/total_words*1.0) * 100.0

def get_doc_bow_features(stories, vocab):
    vocab_size = len(vocab.keys())
    stories_bow = []

    for story in stories:
        story_bow = [0]*vocab_size
        for word in story:
            word_idx = vocab[word]
            story_bow[word_idx] += 1
        stories_bow.append(np.array(story_bow))
    return np.array(stories_bow)

def get_words(vocab, x):
    words = []
    for idx in x:
        for key in vocab:
            if vocab[key] == idx:
                words.append(key)
                break
    return words

def get_reduced_vocab(vocab, word_counts):
    MIN_WORD_THRESHOLD = 10
    for word, count in word_counts.iteritems():
        if count < MIN_WORD_THRESHOLD:
            del vocab[word]

    #REASSIGN INDEXES TO WORDS#
    id = 0
    for word in vocab.keys():
        vocab[word] = id
        id += 1
    vocab['<unk>'] = id
    return vocab

def main_gensim_lda():
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    print("loading corpus...")
    corpus = pickle.load(open("corpus_dev_10samples_annotated_data_nltk_boundaries.pckl",'r'))
    #del corpus.word_ids
    #corpus = clean_corpus(corpus)
    #pickle.dump(corpus,open("corpus_annotated_data_nltk_boundaries.pckl",'w'))

    documents, vocab, word_counts = get_corpus_formatted(corpus)

    stories = get_list_stories_fast(documents,corpus.sent_boundaries,documents.keys(),vocab)

    dict = corpora.Dictionary(stories)

    bow_feats = [dict.doc2bow(story) for story in stories]
    start = timeit.default_timer()
    lda_model = gensim.models.LdaMulticore(corpus=bow_feats,
                                          id2word=dict,
                                          num_topics=100,
                                           #chunksize=1,
                                          passes=50,
                                           workers=15)
    end = timeit.default_timer()

    print (end - start)



    #lda_model.print_topics(100)


def main():


    print("loading corpus...")
    corpus = pickle.load(open("corpus_annotated_data_nltk_boundaries.pckl",'r'))
    #del corpus.word_ids
    #corpus = clean_corpus(corpus)
    #pickle.dump(corpus,open("corpus_annotated_data_nltk_boundaries.pckl",'w'))

    #train_stories,dev_stories, test_stories, vocab = get_train_test_stories(corpus)

    documents, vocab, word_counts = get_corpus_formatted(corpus)
    print("vocab size: "+str(len(vocab.keys())))
    vocab = get_reduced_vocab(vocab, word_counts)
    print("vocab size: "+str(len(vocab.keys())))

    cov = calculate_vocab_coverage(vocab, documents)

    print("coverage: "+str(cov)+"%")
    return

    #train_bow_feats = get_doc_bow_features_fast(documents,
    #                                            corpus.sent_boundaries,
    #                                            corpus.train_set,
    #                                            vocab)
    #test_bow_feats = get_doc_bow_features_fast(documents,
    #                                            corpus.sent_boundaries,
    #                                            corpus.test_set,
    #                                            vocab)
    dev_bow_feats = get_doc_bow_features_fast(documents,
                                                corpus.sent_boundaries,
                                                documents.keys(),
                                                vocab)

    print dev_bow_feats
    model = lda.LDA(n_topics=100, n_iter=1000, random_state=1)
    model.fit(dev_bow_feats)

    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        x = np.argsort(topic_dist)[-10:-1]
        topic_words = get_words(vocab, x)
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))


if __name__ == "__main__":
    main_gensim_lda()
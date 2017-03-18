from gensim import corpora, models
import gensim
import gzip

import os.path
if not os.path.isfile('news.gz'):
    !wget http://acube.di.unipi.it/repo/news.gz

news_corpus = {}
texts = []
categories = set()

with gzip.open('news.gz', 'rb') as news_raw:
    for line in news_raw:
        title = line.lower().strip()
        description = news_raw.readline().lower().strip()
        url = news_raw.readline().lower().strip()
        sequence = news_raw.readline().lower().strip()
        timestamp = news_raw.readline().lower().strip()
        publisher = news_raw.readline().lower().strip()
        category = news_raw.readline().lower().strip()
        news_corpus[sequence] = {'description': description,
                                 'url': url,
                                 'title': title,
                                 'timestamp': timestamp,
                                 'publisher': publisher,
                                 'category': category}
        # To skip the blank line between the records
        news_raw.readline().lower().strip()
        tokens = [token for token in gensim.utils.simple_preprocess(title + b'\n' + description) if token not in gensim.parsing.preprocessing.STOPWORDS]
        texts.append(tokens)
        categories.add(category)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# tfidf = models.tfidfmodel.TfidfModel(corpus)
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=len(categories), id2word = dictionary, passes=30)

tokens = [[dictionary.token2id[word] for word in text] for text in texts]

import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda, corpus, dictionary)

# Saliency related functions adapted from https://github.com/StanfordHCI/termite/blob/master/pipeline/compute_saliency.py
def computeTopicInfo( ldamodel, dictionary ):
    topn = len(dictionary.items())//100 # .1 % of total dictionary terms
    topic_info = []
    for index in range(ldamodel.num_topics):
        topic_weight = sum([prob_score for term, prob_score in ldamodel.show_topic(index, topn=topn) ])
        topic_info.append( {
            'topic' : ldamodel.show_topic(index, topn=topn),
            'weight' : topic_weight
        } )

    return topic_info

def getNormalized( counts ):
    """Rescale a list of counts, so they represent a proper probability distribution."""
    tally = sum( counts )
    if tally == 0:
        probs = [ d for d in counts ]
    else:
        probs = [ d / tally for d in counts ]
    return probs

def computeTermFreq(corpus):
    from collections import Counter
    term_freq = Counter()
    for doc in corpus:
        for (term, freq) in doc:
            term_freq[term] += freq
    return term_freq

def computeTermInfo(ldamodel, dictionary, corpus):
    """Iterate over the list of terms. Compute frequency, distinctiveness, saliency."""
    topic_info = computeTopicInfo(ldamodel, dictionary)
    topic_marginal = getNormalized( [ d['weight'] for d in topic_info ] )
    term_freq = computeTermFreq(corpus)
    term_info = []
    for (tid, term) in dictionary.items():
        # This is not giving expected results or maybe I am using it
        # wrong(?)
        # counts = ldamodel.get_term_topics(tid, minimum_probability=0.00001)
        # if not counts:
        #     print('skippping %s as no term_topics were returned for it' %term)
        #     continue
        # probs = [0 for index in range(ldamodel.num_topics)]
        # for (index, prob) in counts:
        #     probs[index] = prob
        frequency = term_freq[tid]
        probs = []
        for index in range(ldamodel.num_topics):
            probs.append(ldamodel.expElogbeta[index][tid])
        probs = getNormalized( probs )
        distinctiveness = getKLDivergence( probs, topic_marginal )
        saliency = frequency * distinctiveness
        term_info.append({
            'term' : term,
            'saliency' : saliency,
            'frequency' : frequency,
            'distinctiveness' : distinctiveness,
            'rank' : None,
            'visibility' : 'default'
        })
    return term_info

def getKLDivergence( P, Q ):
    """Compute KL-divergence from P to Q"""
    import math
    divergence = 0
    assert len(P) == len(Q)
    for i in range(len(P)):
        p = P[i]
        q = Q[i]
        assert p >= 0
        assert q >= 0
        if p > 0:
            divergence += p * math.log( p / q )
    return divergence

topic_info = computeTopicInfo(ldamodel, dictionary)
term_info = computeTermInfo(ldamodel, dictionary, corpus)

topic_info = sorted( topic_info, key = lambda topic_weight : -topic_weight['weight'] )
term_info = sorted( term_info, key = lambda term_freq : -term_freq['saliency'] )
for i, element in enumerate( term_info ):
    element['rank'] = i

from pipeline.io_utils import *
import os
def CheckAndMakeDirs( path ):
    if not os.path.exists( path ):
        os.makedirs( path )

path = 'data'
SUBFOLDER = 'saliency'
TOPIC_WEIGHTS = 'topic-info.json'
TOPIC_WEIGHTS_TXT = 'topic-info.txt'
TOPIC_WEIGHTS_FIELDS = [ 'term', 'saliency', 'frequency', 'distinctiveness', 'rank', 'visibility' ]
TERM_SALIENCY = 'term-info.json'
TERM_SALIENCY_TXT = 'term-info.txt'
TERM_SALIENCY_FIELDS = [ 'topic', 'weight' ]
    
path = '{}/{}/'.format( path, SUBFOLDER )
CheckAndMakeDirs( path )
# WriteAsJson( term_info, path + TERM_SALIENCY )
# WriteAsJson( topic_info, path + TOPIC_WEIGHTS )
# WriteAsTabDelimited( topic_info, path + TOPIC_WEIGHTS_TXT, TERM_SALIENCY_FIELDS )
    
# for index in range(ldamodel.num_topics):
#     topics = ldamodel.get_topic_terms(index, topn=40)
#     terms = [ldamodel.id2word[topic_id] for (topic_id, prob) in topics]
#     # Reorder terms based on saliency rank
#     reordered_terms = sorted( [term for term in term_info if term['term'] in terms], key = lambda term_freq : term_freq['rank'] )
#     print ('Terms for topic %d order:' %index)
#     print ('{}'.format(' '.join(terms[:20])))
#     print ('{}'.format(' '.join([term['term'] for term in reordered_terms[:20]])))
# print(term_info)

from similarity import *
path = 'data'
path = '{}/{}/'.format( path, SUBFOLDER )
SUBFOLDER = 'similarity'
COMBINED_G2 = 'combined-g2.txt'
combined_g2 = combineSimilarityMatrices(tokens)
CheckAndMakeDirs( path )
# WriteAsSparseMatrix( combined_g2, path + COMBINED_G2 )

import time
from seriation import *

candidateSize = 100
termFreqs = {}
termSaliency = {}
orderedTermList = []
termRank = {}
# termDistinct = {}
#termVisibility = {}
for element in term_info:
    term = dictionary.token2id[element['term']]
    orderedTermList.append( term )
    termSaliency[term] = element['saliency']
    termFreqs[term] = element['frequency']
    termRank[term] = element['rank']
    # termDistinct[term] = element['distinctiveness']
    # termVisibility[term] = element['visibility']

start_time = time.time()
candidateTerms = orderedTermList
term_ordering = []
term_iter_index = []
buffers = [0,0]

preBest = []
postBest = []
DEFAULT_NUM_SERIATED_TERMS = 100

for iteration in range(DEFAULT_NUM_SERIATED_TERMS):
    print("Iteration no. ", iteration)

    addedTerm = 0
    if len(term_iter_index) > 0:
        addedTerm = term_iter_index[-1]
    if iteration == 1:
        (preBest, postBest) = initBestEnergies(combined_g2, addedTerm, candidateTerms)
    (preBest, postBest, bestEnergies) = getBestEnergies(combined_g2, preBest, postBest, addedTerm)
    (candidateTerms, term_ordering, term_iter_index, buffers) = iterate_eff(combined_g2, termRank, termFreqs, termSaliency, candidateTerms, term_ordering, term_iter_index, buffers, bestEnergies, iteration)

seriation_time = time.time() - start_time

print("seriation time: " +  str(seriation_time))
term_ordering, term_iter_index

term_ordering = [dictionary.id2token[term] for term in term_ordering]
term_iter_index = [str(index) for index in term_iter_index]
# term_ordering, term_iter_index = compute()
CheckAndMakeDirs( 'data/seriation/' )
WriteAsList( term_ordering, 'data/seriation/term-ordering.txt' )
WriteAsList( term_iter_index, 'data/seriation/term-iter-index.txt' )

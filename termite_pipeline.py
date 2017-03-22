from gensim import corpora, models
import gensim
import gzip
import requests
import os.path
import shutil

if not os.path.isfile('news.gz'):
    print('Fetching news archive news.gz')
    url = 'http://acube.di.unipi.it/repo/news.gz'
    response = requests.get(url, stream=True)
    with open('news.gz', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

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

print('Done parsing data')
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# tfidf = models.tfidfmodel.TfidfModel(corpus)
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=len(categories), id2word = dictionary, passes=30)
print('Done training LDA Model')

tokens = [[dictionary.token2id[word] for word in text] for text in texts]

# Saliency related functions adapted from https://github.com/StanfordHCI/termite/blob/master/pipeline/compute_saliency.py
from saliency import *
topic_info = computeTopicInfo(ldamodel, dictionary)
term_info = computeTermInfo(ldamodel, dictionary, corpus)

topic_info = sorted( topic_info, key = lambda topic_weight : -topic_weight['weight'] )
term_info = sorted( term_info, key = lambda term_freq : -term_freq['saliency'] )
for i, element in enumerate( term_info ):
    element['rank'] = i

from pipeline.io_utils import *

def CheckAndMakeDirs( path ):
    if not os.path.exists( path ):
        os.makedirs( path )

root = 'docs'
SUBFOLDER = 'saliency'
TOPIC_WEIGHTS = 'topic-info.json'
TOPIC_WEIGHTS_TXT = 'topic-info.txt'
TOPIC_WEIGHTS_FIELDS = [ 'term', 'saliency', 'frequency', 'distinctiveness', 'rank', 'visibility' ]
TERM_SALIENCY = 'term-info.json'
TERM_SALIENCY_TXT = 'term-info.txt'
TERM_SALIENCY_FIELDS = [ 'topic', 'weight' ]
    
path = '{}/{}/'.format( root, SUBFOLDER )
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

print('Done calculating saliecny values for terms')
from similarity import *
SUBFOLDER = 'similarity'
path = '{}/{}/'.format( root, SUBFOLDER )
COMBINED_G2 = 'combined-g2.txt'
combined_g2 = combineSimilarityMatrices(tokens)
CheckAndMakeDirs( path )
print('Done calculating bond energies between terms')
# WriteAsSparseMatrix( combined_g2, path + COMBINED_G2 )

import time
from seriation import *

candidateSize = 100
termFreqs = {}
termSaliency = {}
orderedTermList = []
termRank = {}
# termDistinct = {}
# termVisibility = {}
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
print('Done with seriation step.')
term_ordering, term_iter_index

term_ordering = [dictionary.id2token[term] for term in term_ordering]
term_iter_index = [str(index) for index in term_iter_index]
# term_ordering, term_iter_index = compute()
TERM_ORDERING = 'term-ordering.txt'
TERM_ITER_INDEX = 'term-iter-index.txt'
SUBFOLDER = 'seriation'
path = '{}/{}/'.format( root, SUBFOLDER )
CheckAndMakeDirs( path )
WriteAsList( term_ordering, path + TERM_ORDERING)
WriteAsList( term_iter_index, path + TERM_ITER_INDEX)

term_topic_submatrix = []
term_subindex = []
for term in term_ordering:
    if term in dictionary.token2id:
        term_topic_matrix = []
        for index in range(ldamodel.num_topics):
            term_topic_matrix.append(ldamodel.expElogbeta[index][dictionary.token2id[term]])
        term_topic_submatrix.append(term_topic_matrix)
        term_subindex.append(term)
seriated_parameters = {
    'termIndex' : term_subindex,
    'topicIndex' : [i for i in range(ldamodel.num_topics)],
    'matrix' : term_topic_submatrix
}
SERIATED_PARAMETERS = 'seriated-parameters.json'

term_rank_map = { term: value for value, term in enumerate(term_iter_index)}
term_order_map = { term: value for value, term in enumerate(term_ordering)}
term_saliency_map = { d['term']: d['saliency'] for d in term_info }
term_distinctiveness_map = { d['term'] : d['distinctiveness'] for d in term_info }
filtered_parameters = {
    'termRankMap' : term_rank_map,
    'termOrderMap' : term_order_map,
    'termSaliencyMap' : term_saliency_map,
    'termDistinctivenessMap' : term_distinctiveness_map
}
FILTERED_PARAMETERS = 'filtered-parameters.json'

topic_index = [i for i in range(ldamodel.num_topics)]
term_topic_submatrix = []
term_subindex = []
for term in term_ordering:
    if term in dictionary.token2id:
        term_topic_matrix = []
        for index in range(ldamodel.num_topics):
            term_topic_matrix.append(ldamodel.expElogbeta[index][dictionary.token2id[term]])
        term_topic_submatrix.append(term_topic_matrix)
        term_subindex.append(term)

term_freqs = { d['term']: d['frequency'] for d in term_info }
global_term_freqs = {
    'termIndex' : term_subindex,
    'topicIndex' : topic_index,
    'matrix' : term_topic_submatrix,
    'termFreqMap' : term_freqs
    }
GLOBAL_TERM_FREQS = 'global-term-freqs.json'

SUBFOLDER = 'data'
path = '{}/{}/'.format( root, SUBFOLDER )
CheckAndMakeDirs( SUBFOLDER )
WriteAsJson( seriated_parameters, path + SERIATED_PARAMETERS )
WriteAsJson( filtered_parameters, path + FILTERED_PARAMETERS )
WriteAsJson( global_term_freqs, path + GLOBAL_TERM_FREQS )

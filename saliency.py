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

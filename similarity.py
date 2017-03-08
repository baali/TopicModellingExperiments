# tokenAPI returns dictionary of docID and all the tokens in that
# Document. This is first step in calculating similarity
from collections import Counter
import math

DEFAULT_SLIDING_WINDOW_SIZE = 10
MAX_FREQ = 100.0

def computeDocumentCooccurrence(tokens):
    document_count = 0
    occurrence = Counter()
    cooccurrence = Counter()
    for docID, docTokens in enumerate(tokens):
        tokenSet = frozenset(docTokens)
        document_count += 1
        for token in tokenSet:
            occurrence[token] += 1
        for aToken in tokenSet:
            for bToken in tokenSet:
                # I am not sure what it is comparing here, token ID?
                if aToken < bToken:
                    cooccurrence[(aToken, bToken)] += 1

    return document_count, occurrence, cooccurrence

def getSlidingWindowTokens(tokens, sliding_window_size = DEFAULT_SLIDING_WINDOW_SIZE):
    allWindows = []
    aIndex = 0 - sliding_window_size
    bIndex = len(tokens) + sliding_window_size
    for index in range( aIndex, bIndex ):
        a = max( 0           , index - sliding_window_size )
        b = min( len(tokens) , index + sliding_window_size )
        allWindows.append( tokens[a:b] )
    return allWindows
    
def computeSlidingWindowCooccurrence(tokens, sliding_window_size = DEFAULT_SLIDING_WINDOW_SIZE):
    window_count = 0
    occurrence = Counter()
    cooccurrence = Counter()
    for docID, docTokens in enumerate(tokens):
        allWindowTokens = getSlidingWindowTokens( docTokens, sliding_window_size )
        for windowTokens in allWindowTokens:
            tokenSet = frozenset(windowTokens)
            window_count += 1
            for token in tokenSet:
                occurrence[token] += 1
            for aToken in tokenSet:
                for bToken in tokenSet:
                    if aToken < bToken:
                        cooccurrence[(aToken, bToken)] += 1

    return window_count, occurrence, cooccurrence 

def computeTokenCounts( tokens ):
    token_count = sum( len(docTokens) for docTokens in tokens )

    unigram_counts = Counter()
    for docTokens in tokens:
        for token in docTokens:
            unigram_counts[token] += 1

    bigram_counts = Counter()
    for docTokens in tokens:
        prevToken = None
        for currToken in docTokens:
            if prevToken is not None:
                bigram_counts[(prevToken, currToken)] += 1
            prevToken = currToken

    return token_count, unigram_counts, bigram_counts

def getBinomial( B_given_A, any_given_A, B_given_notA, any_given_notA ):
    assert B_given_A >= 0
    assert B_given_notA >= 0
    assert any_given_A >= B_given_A
    assert any_given_notA >= B_given_notA

    a = float( B_given_A )
    b = float( B_given_notA )
    c = float( any_given_A )
    d = float( any_given_notA )
    E1 = c * ( a + b ) / ( c + d )
    E2 = d * ( a + b ) / ( c + d )

    g2a = 0
    g2b = 0
    if a > 0:
        g2a = a * math.log( a / E1 )
    if b > 0:
        g2b = b * math.log( b / E2 )
    return 2 * ( g2a + g2b )

def getG2( freq_all, freq_ab, freq_a, freq_b ):
    assert freq_all >= freq_a
    assert freq_all >= freq_b
    assert freq_a >= freq_ab
    assert freq_b >= freq_ab
    assert freq_all >= 0
    assert freq_ab >= 0
    assert freq_a >= 0
    assert freq_b >= 0

    B_given_A = freq_ab
    B_given_notA = freq_b - freq_ab
    any_given_A = freq_a
    any_given_notA = freq_all - freq_a

    return getBinomial( B_given_A, any_given_A, B_given_notA, any_given_notA )

def getG2Stats( max_count, occurrence, cooccurrence ):
    g2_stats = {}
    freq_all = max_count
    for ( firstToken, secondToken ) in cooccurrence:
        freq_a = occurrence[ firstToken ]
        freq_b = occurrence[ secondToken ]
        freq_ab = cooccurrence[ (firstToken, secondToken) ]

        scale = MAX_FREQ / freq_all
        rescaled_freq_all = freq_all * scale
        rescaled_freq_a = freq_a * scale
        rescaled_freq_b = freq_b * scale
        rescaled_freq_ab = freq_ab * scale
        if rescaled_freq_a > 1.0 and rescaled_freq_b > 1.0:
            g2_stats[ (firstToken, secondToken) ] = getG2( freq_all, freq_ab, freq_a, freq_b )
    return g2_stats

def combineSimilarityMatrices(tokens):
    combined_g2 = {}

    keys_queued = []
    document_count, document_occurrence, document_cooccurrence = computeDocumentCooccurrence(tokens)
    window_count, window_occurrence, window_cooccurrence = computeSlidingWindowCooccurrence(tokens)
    token_count, unigram_counts, bigram_counts = computeTokenCounts(tokens)
    document_g2 = getG2Stats( document_count, document_occurrence, document_cooccurrence )
    window_g2 = getG2Stats( window_count, window_occurrence, window_cooccurrence )
    collocation_g2 = getG2Stats( token_count, unigram_counts, bigram_counts )
    for key in document_g2:
        ( firstToken, secondToken ) = key
        otherKey = ( secondToken, firstToken )
        keys_queued.append( key )
        keys_queued.append( otherKey )
    for key in window_g2:
        ( firstToken, secondToken ) = key
        otherKey = ( secondToken, firstToken )
        keys_queued.append( key )
        keys_queued.append( otherKey )
    for key in collocation_g2:
        keys_queued.append( key )

    keys_processed = {}
    for key in keys_queued:
        keys_processed[ key ] = False

    for key in keys_queued:
        if not keys_processed[ key ]:
            keys_processed[ key ] = True

            ( firstToken, secondToken ) = key
            if firstToken < secondToken:
                orderedKey = key
            else:
                orderedKey = ( secondToken, firstToken )
            score = 0.0
            if orderedKey in document_g2:
                score += document_g2[ orderedKey ]
            if orderedKey in window_g2:
                score += window_g2[ orderedKey ]
            if key in collocation_g2:
                score += collocation_g2[ key ]
            if score > 0.0:
                combined_g2[ key ] = score
    return combined_g2

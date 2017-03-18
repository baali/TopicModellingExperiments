from operator import itemgetter
candidateSize = 100
def initBestEnergies(combined_g2, firstTerm, candidateTerms):
    preBest = []
    postBest = []
    for candidate in candidateTerms:
        pre_score = 0
        post_score = 0

        # preBest
        if (candidate, firstTerm) in combined_g2:
            pre_score = combined_g2[(candidate, firstTerm)]
        # postBest
        if (firstTerm, candidate) in combined_g2:
            post_score = combined_g2[(firstTerm, candidate)]

        preBest.append((candidate, pre_score))
        postBest.append((candidate, post_score))

    return (preBest, postBest)

def getBestEnergies(combined_g2, preBest, postBest, addedTerm):
    if addedTerm == 0:
        return (preBest, postBest, [])

    term_order = [x[0] for x in preBest]
    # compare candidate terms' bests against newly added term
    remove_index = -1
    for existingIndex in range(len(preBest)):
        term = term_order[existingIndex]
        if term == addedTerm:
            remove_index = existingIndex

        # check pre energies
        if (term, addedTerm) in combined_g2:
            if combined_g2[(term, addedTerm)] > preBest[existingIndex][1]:
                preBest[existingIndex] = (term, combined_g2[(term, addedTerm)])
        # check post energies
        if (addedTerm, term) in combined_g2:
            if combined_g2[(addedTerm, term)] > postBest[existingIndex][1]:
                postBest[existingIndex] = (term, combined_g2[(addedTerm, term)])

    # remove the added term's preBest and postBest scores
    if remove_index != -1:
        del preBest[remove_index]
        del postBest[remove_index]

    #create and sort the bestEnergies list
    energyMax = [sum(pair) for pair in zip([x[1] for x in preBest], [y[1] for y in postBest])]
    bestEnergies = zip([x[0] for x in preBest], energyMax)

    return (preBest, postBest, sorted(bestEnergies, key=itemgetter(1), reverse=True))

def iterate_eff( combined_g2, termRank, termFreqs, termSaliency, candidateTerms, term_ordering, term_iter_index, buffers, bestEnergies, iteration_no ):
    maxEnergyChange = 0.0;
    maxTerm = "";
    maxPosition = 0;

    if len(bestEnergies) != 0:
        bestEnergy_terms = [x[0] for x in bestEnergies]
    else:
        bestEnergy_terms = candidateTerms

    breakout_counter = 0
    for candidate_index in range(len(bestEnergy_terms)):
        breakout_counter += 1
        candidate = bestEnergy_terms[candidate_index]
        for position in range(len(term_ordering)+1):
            current_buffer = buffers[position]
            candidateRank = termRank[candidate]
            if candidateRank <= (len(term_ordering) + candidateSize):
                current_energy_change = getEnergyChange(combined_g2, termFreqs, termSaliency, candidate, position, term_ordering, current_buffer, iteration_no)
                if current_energy_change > maxEnergyChange:
                    maxEnergyChange = current_energy_change
                    maxTerm = candidate
                    maxPosition = position
                    # check for early termination
        if candidate_index < len(bestEnergy_terms)-1 and len(bestEnergies) != 0:
            if maxEnergyChange >= (2*(bestEnergies[candidate_index][1] + current_buffer)):
                print("#-------- breaking out early ---------#")
                print("candidates checked: ", breakout_counter)
                break;

    print("change in energy: ", maxEnergyChange)
    print("maxTerm: ", maxTerm)
    print("maxPosition: ", maxPosition)

    candidateTerms.remove(maxTerm)

    # update buffers
    buf_score = 0
    if len(term_ordering) == 0:
        # what a waaat?
        buffers = buffers
    elif maxPosition >= len(term_ordering):
        if (term_ordering[-1], maxTerm) in combined_g2:
            buf_score = combined_g2[(term_ordering[-1], maxTerm)]
        buffers.insert(len(buffers)-1, buf_score)
    elif maxPosition == 0:
        if (maxTerm, term_ordering[0]) in combined_g2:
            buf_score = combined_g2[(maxTerm, term_ordering[0])]
        buffers.insert(1, buf_score)
    else:
        if (term_ordering[maxPosition-1], maxTerm) in combined_g2:
            buf_score = combined_g2[(term_ordering[maxPosition-1], maxTerm)]
        buffers[maxPosition] = buf_score

        buf_score = 0
        if (maxTerm, term_ordering[maxPosition]) in combined_g2:
            buf_score = combined_g2[(maxTerm, term_ordering[maxPosition])]
        buffers.insert(maxPosition+1, buf_score)

    # update term ordering and ranking
    if maxPosition >= len(term_ordering):
        term_ordering.append(maxTerm)
    else:
        term_ordering.insert(maxPosition, maxTerm)
    term_iter_index.append(maxTerm)
    return (candidateTerms, term_ordering, term_iter_index, buffers)

def getEnergyChange(combined_g2, termFreqs, termSaliency, candidateTerm, position, term_list, currentBuffer, iteration_no):
    prevBond = 0.0
    postBond = 0.0

    # first iteration only
    if iteration_no == 0:
        current_freq = 0.0
        current_saliency = 0.0

        if candidateTerm in termFreqs:
            current_freq = termFreqs[candidateTerm]
        if candidateTerm in termSaliency:
            current_saliency = termSaliency[candidateTerm]
        return 0.001 * current_freq * current_saliency

    # get previous term
    if position > 0:
        prev_term = term_list[position-1]
        if (prev_term, candidateTerm) in combined_g2:
            prevBond = combined_g2[(prev_term, candidateTerm)]

    # get next term
    if position < len(term_list):
        next_term = term_list[position]
        if (next_term, candidateTerm) in combined_g2:
            postBond = combined_g2[(candidateTerm, next_term)]

    return 2*(prevBond + postBond - currentBuffer)

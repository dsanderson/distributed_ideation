#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:02 2017

@author: arlittr
"""

from spacy.en import English
parser = English()

import pandas as pd
from nltk.corpus import stopwords as stopwords
import networkx as nx
import string
import matplotlib.pyplot as plt

def cleanPassage(rawtext):
    #some code from https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    
    #if data is bad, return empty
    if type(rawtext) is not str:
        return ''
    
    #split text with punctuation
    bad_chars = "".join(string.punctuation)
    for c in bad_chars: rawtext = rawtext.replace(c, "")
    
    #parse 
    tokens = parser(rawtext)

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    
    return tokens

def getLemmas(tokens):
    # lemmatize
    lemmas = [tok.lemma_.lower().strip() for tok in tokens]
#    lemmas = []
#    for tok in tokens:
#        lemmas.append(tok.lemma_.lower().strip())
    return lemmas

def makeNodelist(tokens):
    BADPOS = ['PUNCT','NUM','X']
    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\\"]
    nodes = []
    for tok in tokens:
        goodPOS = tok.pos_ not in BADPOS 
        notStopword = tok.orth_ not in stopwords.words('english')
        notSymbol= tok.orth_ not in SYMBOLS
        isMeaningful = tok.prob > probs_cutoff
        if goodPOS and notStopword and notSymbol and isMeaningful:
            nodes.append(tok.lemma_+' '+tok.pos_+' '+str(tok.prob))
#            nodes.append(tok.lemma_)
    return nodes

def findMeaningfulCutoffProbability(alltokens):
    probs = [tok.prob for tok in alltokens]
    probs.sort()
    plt.plot(probs)
    plt.xlabel('Rank')
    plt.ylabel('Log Probability')
    plt.grid()
    plt.show()
    #set probs_cutoff by inspection by looking for the elbow on the plot of sorted log probabilities
#    probs_cutoff = 500
#    probs_cutoff = probs[int(input("By inspection, at which rank is the elbow for the log probability plot? [integer]"))]
    
    #removing the lowest observed probability seems to remove most of the spelling errors
    probs_cutoff = min(probs)
    return probs_cutoff


def buildNetwork(nodesLOL):
    #http://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
    
    G = nx.Graph()
    for nodeslist in nodesLOL:
        Gnew=nx.complete_graph(len(nodeslist))
        nx.relabel_nodes(Gnew,dict(enumerate(nodeslist)), copy=False)
        G = nx.compose(G,Gnew)   
    
    return G

  

        
if __name__ == '__main__':
    #Read descriptions of concepts (or read in words)
    path = '/Volumes/SanDisk/01 Data and Analysis/original data/Grouped & Novice/Grouped & Novice - RAW Data.csv'
    gn = pd.read_csv(path)#, usecols=['Text: Verbatim','Interpretation (Entries without words - Raw data contain only words)'])
    gn.rename(columns={'Text: Verbatim':'raw'},inplace=True)
    
    #Split into data structure (if necessary)
    
    #clean passages
    gn['tokens'] = gn['raw'].apply(lambda x: cleanPassage(x))
#    gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))
    
    probs_cutoff = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])
    gn['nodeslist'] = gn['tokens'].apply(lambda tok: makeNodelist(tok))

    
    #build network
    G_gn = buildNetwork([n for n in gn['nodeslist']])
    
    #calculate modularity
    community_growth = []
    max_cliques = 20
    for k in range(max_cliques):
        community_growth.append(len(list(nx.k_clique_communities(G_gn,k+2))))
    plt.plot([k for k in range(2,max_cliques+1+2)],community_growth,'*')
    plt.xlabel('clique size (k)')
    plt.ylabel('Qty Clique Communities')
    plt.title('Number of cliques vs clique size via percolation method')
    plt.grid()
    plt.show()
    
    #write out gephi file for visualization
    nx.write_graphml(G_gn,'gn.graphml')
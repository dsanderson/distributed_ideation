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
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import numpy as np
from math import inf

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
    return lemmas

def makeNodelist(row):
    tokens = row['tokens']
    uid = row['uid']
    BADPOS = ['PUNCT','NUM','X','SPACE']
    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
    nodes = []
    for tok in tokens:
        goodPOS = tok.pos_ not in BADPOS 
        notStopword = tok.orth_ not in stopwords.words('english')
        notSymbol = tok.orth_ not in SYMBOLS
        isMeaningful = tok.prob > probs_cutoff
        if goodPOS and notStopword and notSymbol and isMeaningful:
            nodes.append(tok.lemma_+' '+tok.pos_)
    return nodes

def makeNodelist2(tokens):
    BADPOS = ['PUNCT','NUM','X','SPACE']
    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
    nodes = []
    for tok in tokens:
        goodPOS = tok.pos_ not in BADPOS 
        notStopword = tok.orth_ not in stopwords.words('english')
        notSymbol = tok.orth_ not in SYMBOLS
        isMeaningful = tok.prob > probs_cutoff
        if goodPOS and notStopword and notSymbol and isMeaningful:
            nodes.append(tok.lemma_+' '+tok.pos_)
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


def buildNetwork(nodesLOL,attr={}):
    #http://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
    #If we have the same word repeated anywhere, we only make one node for it
    #TODO: add attribute that tracks UIDs for every source that a node appears in
    G = nx.Graph()
    for nodeslist in tqdm(nodesLOL,desc='Bulding Network'):
        Gnew=nx.complete_graph(len(nodeslist))
        nx.relabel_nodes(Gnew,dict(enumerate(nodeslist)),copy=False)
        for attrname,attrmappings in attr.items():
            nx.set_node_attributes(Gnew,attrname,attrmappings)
        G = nx.compose(G,Gnew)   
    return G

def plotPgvGraph(G,filename=None,printRelationships=None,promoteNodeLabels=None):
    #network plotting from old IBFM project
    G2 = G.copy()
    
    #print relationships on edges
    if printRelationships:
        for n,nbrs in G2.adjacency_iter():
            for nbr in nbrs.keys():
                for edgeKey,edgeProperties in G2[n][nbr].items():
                    G2[n][nbr][edgeKey]['label'] = edgeProperties[printRelationships]
                    
    #promote the attribute in promoteNodeLabels to node label
    if promoteNodeLabels:
        for n in G2.nodes_iter():
            try:
                G2.node[n]['label'] = G2.node[n][promoteNodeLabels]
            except:
                G2.node[n]['label'] = None
    
    #draw graph
    thisG = nx.drawing.nx_pydot.to_pydot(G2)
    
    if filename==None:
        filename = 'plots/'+ 'junk' + '.svg'
    thisG.write(filename,format='svg')  

def dumpDataFrame():
    return 0
        
if __name__ == '__main__':
    #Read descriptions of concepts (or read in words)

    basepath = '/Volumes/SanDisk/01 Data and Analysis/original data/Grouped & Novice/'
    basename = 'Grouped & Novice - '
    namespecifier = 'RAW Data'
    fileextension = '.csv'
#    path = basepath + basename + namespecifier + fileextension
#    path = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_data.csv'
    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/Distributed Experience and Novice (superset) clean subset.csv'
#    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/Distributed Experience and Novice (superset) clean 2.csv'
#    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/test.csv'
    gn = pd.read_csv(path)#,sep='\t')#,error_bad_lines=False)
    gn.rename(columns={'Text: Verbatim':'raw','Image No':'uid'},inplace=True)
        
    #clean passages
    tqdm.pandas(desc="Make Spacy Tokens")
    gn['tokens'] = gn['raw'].progress_apply(lambda x: cleanPassage(x))
    gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))
    
    probs_cutoff = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])
#    gn['nodeslist'] = gn.apply(makeNodelist,axis=1)
    tqdm.pandas(desc="Make Nodeslist")
    gn['nodeslist'] = gn['tokens'].progress_apply(lambda x: makeNodelist2(x))

    print('Done making nodelist')
    
    #%%
    #build network
#    iddict = {index:row['nodeslist'] for index,row in gn.iterrows()}
    G_gn = buildNetwork([n for n in gn['nodeslist']])#,{'conceptid':{row['nodeslist']:index for index,row in gn.iterrows()})   # {m:gn[ for n in gn['nodeslist'] for m in n}})
    print('Done making network')
    #add attributes to nodes
    #concept id (from pandas table row?)
    #%%
    
    
    #calculate modularity
    community_growth = []
    max_cliques = 20
    for k in tqdm(range(max_cliques),desc='Running k-clique modularity algorithm'):
        community_growth.append(len(list(nx.k_clique_communities(G_gn,k+2))))
    plt.plot([k for k in range(2,max_cliques+2)],community_growth,'*')
    plt.xlabel('clique size (k)')
    plt.ylabel('Qty Clique Communities')
    plt.title('Number of k-clique communities vs clique size via percolation method')
    plt.grid()
    plt.show()
    
    #%%
    #write out gephi file for visualization
    nx.write_graphml(G_gn,'gn.graphml')
    
    #write out dataframe for posterity
    outpath = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_processed.csv'
    outpath = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_processed.csv'
    gn.to_csv(outpath,encoding='utf-8')
    
    #%%
    full_distance_matrix = nx.floyd_warshall_numpy(G_gn)
    fmax = np.finfo(np.float64).max
    full_distance_matrix[full_distance_matrix==inf] = fmax
    db = DBSCAN(eps=1.0,min_samples=100,metric='precomputed').fit(full_distance_matrix)
    print(db.labels_)
    print(db.components_)
    
#    outpath = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_fig.svg'
#    plotPgvGraph(G_gn,outpath)
    #----maybe move this later, just taking notes right now----
    #assign a module ID to each node
    
    
    #compare module assignments between manual ratings and automatic ratings
    #current idea: greedily assign manual to automatic by calculating jaccard sim
        #assign, remove from set, repeat
        #measure of performance = minimize(avg(all jaccard similarities))
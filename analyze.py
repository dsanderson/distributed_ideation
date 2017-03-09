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
import numpy as np
from math import inf
import hdbscan
from datetime import datetime


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


def makeNodelist(tokens):
#    BADPOS = ['PUNCT','NUM','X','SPACE']
    GOODPOS = ['NOUN','PROPN','VERB','ADJ','ADV']
    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
    probs_cutoff_upper = -7.6 #by inspection of sample data
    nodes = []
    for tok in tokens:
        goodPOS = tok.pos_ in GOODPOS 
        notStopword = tok.orth_ not in stopwords.words('english')
        notSymbol = tok.orth_ not in SYMBOLS
        isMeaningful = tok.prob > probs_cutoff_lower and tok.prob < probs_cutoff_upper
        
        if goodPOS and notStopword and notSymbol and isMeaningful:
            nodes.append(tok.lemma_+' '+tok.pos_)
    return nodes

#def makeNodelist2(tokensuid):
#    
##    BADPOS = ['PUNCT','NUM','X','SPACE']
#    GOODPOS = ['NOUN','PROPN','VERB','ADJ','ADV']
#    SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
#    probs_cutoff_upper = -7.6 #by inspection of sample data
#    nodes = []
#    for tok in tokens:
#        goodPOS = tok.pos_ in GOODPOS 
#        notStopword = tok.orth_ not in stopwords.words('english')
#        notSymbol = tok.orth_ not in SYMBOLS
#        isMeaningful = tok.prob > probs_cutoff_lower and tok.prob < probs_cutoff_upper
#        
#        if goodPOS and notStopword and notSymbol and isMeaningful:
#            nodes.append(tok.lemma_+' '+tok.pos_)
#    return nodes        

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
    probs_cutoff_lower = min(probs)
    return probs_cutoff_lower


def buildNetwork(nodesLOL,attr={}):
    #http://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
    #If we have the same word repeated anywhere, we only make one node for it
    G = nx.Graph()
    for nodeslist in tqdm(nodesLOL,desc='Bulding Network'):
        Gnew=nx.complete_graph(len(nodeslist))
        nx.relabel_nodes(Gnew,dict(enumerate(nodeslist)),copy=False)
        if attr:
            this_attr = {k:None for k in attr.keys()}
            for key in this_attr.keys():
                this_attr[key] = {k:attr[key][k] for k in nodeslist}
            for attrname,attrmappings in this_attr.items():
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

def calculateModularity():
    #very slow!
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

def dumpTokenProbabilities(tokens,path):
    print('Dumping token probabilities ' + str(datetime.now()))
    sorted_tokens = sorted([(t.prob,t) for t in tokens],key=lambda tup: tup[0])
    probs_dict = {'probs':[e[0] for e in sorted_tokens],'words':[e[1] for e in sorted_tokens]}
    probs_results = pd.DataFrame(probs_dict)
    probs_results.to_csv(path)
    
def generateAttributesDict(tokens,uids,nodeslist):
    d = {}
    d['uid'] = {}
    d['token'] = {}
    for token,uid,nodelist in zip(tokens,uids,nodeslist):
        for node,t in zip(nodelist,token):
            try:
                d['uid'][node].append(uid)
            except:
                d['uid'][node] = [uid]
            try:
                d['token'][node].append(t)
            except:
                d['token'][node] = [t]
    return d

if __name__ == '__main__':
    #Read descriptions of concepts (or read in words)
    inputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/input_data/'
    outputbasepath = '/Volumes/SanDisk/Repos/distributed_ideation/results/'
#    basename = 'Distributed Experience and Novice (superset) clean'
    basename = 'Distributed Experience and Novice (superset) clean TEST SAMPLE'
#    basename = 'Group Experienced first and second round (unique set) clean'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
#    path = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_data.csv'
#    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/Distributed Experience and Novice (superset) clean subset.csv'
#    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/Distributed Experience and Novice (superset) clean 2.csv'
#    path = '/Volumes/SanDisk/01 Data and Analysis/raw data/test.csv'

    
    gn = pd.read_csv(path)
    gn.rename(columns={'Text: Verbatim':'raw','Unique ID':'uid'},inplace=True)
        
    #clean passages
    tqdm.pandas(desc="Make Spacy Tokens")
    gn['tokens'] = gn['raw'].progress_apply(lambda x: cleanPassage(x))
    gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))
    
    probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])
    tqdm.pandas(desc="Make Nodeslist")
    gn['nodeslist'] = gn['tokens'].progress_apply(lambda x: makeNodelist(x))
    
    #Generate attributes for nodes in the graph
    nodeAttributesDict = generateAttributesDict(gn.tokens,gn.uid,gn.nodeslist)

    print('Done making nodelist ' + str(datetime.now()))
    
    path = outputbasepath + basename + ' word probabilities.csv'
    dumpTokenProbabilities([t for tok in gn['tokens'] for t in tok],path)
    #%%
    #build network
#    iddict = {index:row['nodeslist'] for index,row in gn.iterrows()}
    G_gn = buildNetwork([n for n in gn['nodeslist']],nodeAttributesDict)#,{'conceptid':{row['nodeslist']:index for index,row in gn.iterrows()})   # {m:gn[ for n in gn['nodeslist'] for m in n}})
    print('Done making network ' + str(datetime.now()))
    #add attributes to nodes
    #concept id (from pandas table row?)
    #%%
    #calculate modularity
    
    print('Calculating Modularity ' + str(datetime.now()))
#    calculateModularity()
    
    #%%
    #write out gephi file for visualization
    print('Writing graphml file ' + str(datetime.now()))
    outpath = outputbasepath+basename+' graphml'+'.graphml'
    #need copy because nx.write_graphml cant handle spacy tokens
    #build attributeless copy because spacy tokens break G.copy()
    G_gn_shallow = buildNetwork([n for n in gn['nodeslist']])
    nx.write_graphml(G_gn_shallow,outpath)
    
    #write out dataframe for future analysis
    print('Writing out dataframe ' + str(datetime.now()))
    outpath = outputbasepath+basename+' dataframe'+'.csv'
    gn.to_csv(outpath,encoding='utf-8')
    
    #%%
    print('Generating distance matrix ' + str(datetime.now()))
    full_distance_matrix = nx.floyd_warshall_numpy(G_gn)
    fmax = np.finfo(np.float64).max
    full_distance_matrix[full_distance_matrix==inf] = fmax
    outpath = outputbasepath+basename+' distance matrix'+'.csv'
    np.savetxt(outpath,full_distance_matrix,delimiter=",")
    
    #%% Cluster with HDBSCAN- takes the guesswork out of determining similarity parameter
    print('Clustering with HDBSCAN ' + str(datetime.now()))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(full_distance_matrix)
    
    #%% Write out clustering results
    print('Writing out clustering results ' + str(datetime.now()))
    clustering_results_d = {'nodes':G_gn.nodes(),
                            'clusters':cluster_labels,
                            'uids':[nx.get_node_attributes(G_gn,'uid')[n] for n in G_gn.nodes()]
                            }
    clustering_results = pd.DataFrame(clustering_results_d)
    clustering_results['nodeDegree'] = clustering_results['nodes'].apply(lambda x: G_gn.degree(x))
    clustering_results['frequency'] = clustering_results['uids'].apply(lambda x: len(x))
    outpath = outputbasepath+basename+' clustering results'+'.csv'
    clustering_results.to_csv(outpath,encoding='utf-8')
    
    #%%
    
#    outpath = '/Volumes/SanDisk/01 Data and Analysis/01 network anlysis - gephi/test_graph_fig.svg'
#    plotPgvGraph(G_gn,outpath)
    #----maybe move this later, just taking notes right now----
    #assign a module ID to each node
    
    
    #compare module assignments between manual ratings and automatic ratings
    #current idea: greedily assign manual to automatic by calculating jaccard sim
        #assign, remove from set, repeat
        #measure of performance = minimize(avg(all jaccard similarities))
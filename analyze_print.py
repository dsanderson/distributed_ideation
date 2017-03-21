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
from tqdm import tqdm
import numpy as np
from math import inf
from datetime import datetime
import sys
import statistics
import random
import os

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
    tokens = [tok for tok in tokens if str(tok) not in stopwords.words('english')]
    return tokens

def getLemmas(tokens):
    # lemmatize
    lemmas = [tok.lemma_.lower().strip() for tok in tokens]
    return lemmas



def makeNodelist(tokens,limitPOS=None):
#    BADPOS = ['PUNCT','NUM','X','SPACE']
    if limitPOS:
        GOODPOS = limitPOS
    else:
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

def findMeaningfulCutoffProbability(alltokens):
    probs = [tok.prob for tok in alltokens]
    probs.sort()
    probs_cutoff_lower = min(probs)
    return probs_cutoff_lower


def buildNetwork(nodesLOL,attr={}):
    #http://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
    #If we have the same word repeated anywhere, we only make one node for it
    G = nx.Graph()
    for nodeslist in nodesLOL:
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

def calculateModularity(G_gn, max_cliques=20):
    #very slow!, exp(N) complexity
    communities = list(nx.k_clique_communities(G_gn,max_cliques+2)) #TODO Ask what the +2 is for
    return communities

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
    #keep only unique uids
    for node,uidlist in d['uid'].items():
        d['uid'][node] = list(set(d['uid'][node]))
    return d

def save_clusters(clusters, G_gn, path):
    with open(path,'w') as f:
        for i in range(0, len(clusters)):
            for j in clusters[i]:
                #f.write(token+','+str(i)+'\n')
                degree = G_gn.degree(j)
                ideas = G_gn.node[j]['uid']
                s_ideas = ','.join([str(k) for k in ideas])
                l_ideas = len(ideas)
                print(j+','+str(i)+','+str(degree)+','+str(l_ideas)+','+s_ideas+'\n')

if __name__ == '__main__':
    #Read descriptions of concepts (or read in words)
    inputbasepath = '/'
    outputbasepath = '/'
    basename = 'samples'
#     basename = 'Distributed Experience and Novice (superset) clean TEST SAMPLE'
#    basename = 'Group Experienced first and second round (unique set) clean'
    fileextension = '.csv'
    path = inputbasepath + basename + fileextension
    #Add test to check whether dataframe already exists as a file
    #Then we can just read that in instead of processing raw again
    
    g_all = pd.read_csv(path) # add encoding for windows
    g_all.rename(columns={'Text: Verbatim':'raw','Unique ID':'uid'},inplace=True)
    samples = []

    random.seed(os.environ['SEED']) 
 
    #resample g_all into gn
    gn = g_all.sample(int(os.environ['SIZE']))
    #clean passages
    gn['tokens'] = gn['raw'].apply(lambda x: cleanPassage(x))
    gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))
    probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])
    gn['nodeslist'] = gn['tokens'].apply(lambda x: makeNodelist(x))
    #Generate attributes for nodes in the graph
    nodeAttributesDict = generateAttributesDict(gn.tokens,gn.uid,gn.nodeslist)
    #print('Done making nodelist ' + str(datetime.now()))
    
    path = outputbasepath + basename + ' word probabilities.csv'
    dumpTokenProbabilities([t for tok in gn['tokens'] for t in tok],path)

    G_gn = buildNetwork([n for n in gn['nodeslist']],nodeAttributesDict)
    #print('Done making network ' + str(datetime.now()))
    
    #print('Calculating Modularity ' + str(datetime.now()))
    max_cliques = int(os.environ['K'])
    modules = calculateModularity(G_gn, max_cliques)
    #print(G_gn.node[list(modules[0])[0]]['uid'])
    #print(G_gn.degree(list(modules[0])[0]))
    #print(modules)
    save_clusters(modules,G_gn,"clusters.csv")

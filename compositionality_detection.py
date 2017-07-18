import os
import math
import numpy as np
from pca import PCA
from nltk import word_tokenize
import string
import pickle

dict_path = "vector.pickle"
with open(dict_path, "rb") as handle:
    vecDict, dim = pickle.load(handle)
print "loading dictionary..."

def loadVecAsDict(word_vec_fn, dict_path):
    """
    load word embedding as a dictionary
    """
    vf = open(word_vec_fn,"r")
    vecDict = dict()
    vf.readline()
    for line in vf:
        wordVec = line.strip().split()
        word = wordVec.pop(0)
        dim = len(wordVec)
        wordVec = np.array(wordVec, "float")
        vecDict[word] = wordVec[:]
    with open(dict_path, "wb") as handle:
        pickle.dump((vecDict, dim), handle)
    print "finish saving word embedding ..."
    

def filterContext(sen):
    """
    given a sentence (no punctuation), filter the functional words
    return a string consisting of content words
    """
    fwList = ['e','.g.','etc','the','a','an','which','is','are','be','will','and','it','they',"'s","one's",'before','so',',',';','.','something','anything','that','cannot']
    sen = " "+sen+" "
    for fw in fwList:
        sen = sen.replace(" "+fw+" "," ")
    # tokenization
    sen_tok = word_tokenize(sen)
    # punctuations
    punc = list(string.punctuation)
    punc.append("''")
    sen_tok_punc = [i for i in sen_tok if i not in punc]
    sen_new = " ".join(sen_tok_punc)
    return sen_new.strip()


def readContext(w, sent, winSize):
    """
    extract words around w within a cxt_window from sent
    return: a list of context words
    """
    wordSeq = sent.split()
    wordCount  = len(wordSeq)
    context= []
    for i in range(wordCount):
        word = wordSeq[i]
        if (word == w):
            for j in range(-winSize, winSize+1):
                # do not include phrase iteself
                if (j == 0):
                    continue
                    #print "current word"
                else:
                    if ((i+j < 0) or (i+j)> wordCount -1):
                        print "sentence is not long"
                    else:
                        context.append(wordSeq[i+j])
            break
    return context


def getPhraseEmbed(wl, dim):
    """
    sum component word embeddings as phrase embedding
    wl: a list of component words
    dim: word dimension
    """
    senEmbed = np.zeros(dim)
    for word in wl:
        if (word not in vecDict):
            print "non-exist:", word
            continue
        wordEmbed = np.array(vecDict[word])
        senEmbed = senEmbed + wordEmbed
    return senEmbed

def getCxtSubspace(wl, dim, var_threshold=0.45):
    emb = []
    for word in wl:
        if (word not in vecDict):
            print "non-exist:", word
            continue
        wordEmbed = vecDict[word]
        emb.append(wordEmbed)
    emb = np.array(emb)
    
    pca =PCA()
    pca.fit(emb)
    varList = pca.explained_variance_ratio_
    cand = 0
    varSum = 0
    for var in varList:
        varSum += var
        cand += 1
        if (varSum >= var_threshold):
            break

    pca= PCA(n_components=cand)
    pca.fit(emb)
    top_embed = pca.components_
    print "dim:", len(top_embed.tolist()), cand
    return top_embed.tolist()

def getSimilarity(a, b):
    if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        return 0
    return 1.0 * np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def getCxtLinCom(X,w):
    """
    X: context subspace
    w: compositional phrase/word embedding
    return: cosine similarity between w and w_appro (w's projection to subspace X)
    """

    X = np.asarray(X)
    mat = np.dot(X, np.transpose(X))
    w = np.asarray(w)
    col = np.dot(X,w)
    coef = np.linalg.solve(mat,col)
    w_appro = np.dot(np.transpose(X),coef)
    similarity = getSimilarity(w,w_appro)
    return similarity

def scoreCompositionality(line, cxt_window):
    """
    line format: phrase \t sentence
    return compositionality of the phrase in its sentence
    """
    phrase, sent = line.split('\t')
    phrase_word = phrase.replace(" ", "_")
    sent = filterContext(sent)
    sent = sent.replace(phrase, phrase_word)
    phrase_vec = getPhraseEmbed(phrase.split(), dim)
    cxt = readContext(phrase_word, sent, cxt_window)
    cxt_subspace = getCxtSubspace(cxt, dim, var_threshold=0.45)
    score = getCxtLinCom(cxt_subspace, phrase_vec)
    return score

if __name__=="__main__":

    """
    #save dictionary
    loadVecAsDict(word_vec_fn="../vectors_test.txt", dict_path="vector.pickle")
    """
    # below is an example about how to get phrasal compositionality score in a sentence
    # line format: phrase \t sentence
    # compositionality score is [0, 1]
    cxt_window = 12
    compo_line = "blue sky	napoleon stood with his marshals around him it was quite light above him was a clear blue sky and the sun vast orb quivered like a huge hollow crimson float on the surface of that milky sea of mist"
    idiom_line = "blue sky	unrealistic or impractical the author shows what is testable physics, what is blue sky nonsense, not limited by conventional notions of what is practical or feasible and what is philosophy domain"
    compo_score = scoreCompositionality(compo_line, cxt_window)
    idiom_score = scoreCompositionality(idiom_line, cxt_window)
    print "compo_score: %f, idiom_score: %f" % (compo_score, idiom_score)

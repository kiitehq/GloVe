import re 
import nltk 
import argparse
import subprocess
import os

os.chdir("/Users/mandygu/Desktop/GloVe/")

# Preprocessing function which removes non-alphabetical words and tokenizes
def preproc(path):
    print("Preprocessing vocab ... ")
    with open(path, "r") as myfile: 
        word = myfile.read()
    word = word.lower()
    word = re.sub('\[[^\]]*\]','',word)                             # removing footnotes (applicable for wikipedia articles) 
    tokens = nltk.word_tokenize(word)
    tokens = [x for x in tokens if x.isalpha()]
    tokens = " ".join(tokens)
    with open("preprocessed_vocab.txt", "w") as text_file:
        text_file.write(tokens)

# call GloVe shell script which turns tokens to embeddings 
# this turns each token into a vector 

def call(): 
    print("Building embeddings ...")
    subprocess.call(['./demo.sh'])

# Merge custom embeddings with pre-trained embeddings by concatenating the new vector after the vector of the pretrained embeddings 
# TODO: shuffle the embeddings to randomize coordinates (not currently needed as Mnemonic Reader does this already)

def merge(originalEmbeddings): 
    original = [] 
    for i in open(originalEmbeddings, "r"): 
        original.append(i)

    vocab = {}

    n = 0

    for i in open("preprocessed_vocab.txt", "r"):
        i = i.replace(",", "")
        lst = i.split(" ")
        vocab[lst[0]] = lst[1:]
        if n == 0: 
            n = len(lst)

    newEmbeddings = [] 

    for i in original:                              # iterate through the original vocabulary 
        lst = i.split(" ")
        if lst[0].isalpha() == False:               # keep only tokens that are alphabetical      
            continue
        elif lst[0].lower() in vocab.keys():        # if the token is also present in the new vocabulary, append vector from new vocabulary after vector from original vocabulary
            lst[-1] = lst[-1][:-1]  
            lst = lst + vocab[lst[0].lower()]
            newEmbeddings.append(" ".join(lst))
        else:                                       # if the token is not found in the new vocabulary, append an 0 vector after vector from original vocabulary 
            temp = ' 0' * (n-2) + ' 0\n'
            newEmbeddings.append(i[:-1] + temp)
    
    # export document
    export = open("combined_embeddings.txt", "w")

    for item in newEmbeddings:
        export.write("%s" % item)

# removes intermediate file created as a part of the process

def remove(): 
    os.remove("preprocessed_vocab.txt")

# ---------------
#  Command line
# ---------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="custom_vocab.txt", help='name of vocabulary file')
    parser.add_argument('--refine', type = bool, default = True, help = 'do you want to merge with pretrained embeddings? If True, specify')
    parser.add_argument('--merge', type = str, default = "glove.840B.300d.txt", help = 'filename of additional embeddings')
    args = parser.parse_args()

preproc(args.file)
call() 

if args.refine: 
    print("merging embeddings ...")
    merge(args.merge)

remove() 

# -------------------
#  Testing Embeddings
# -------------------

'''

Testing: 

model.most_similar(positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None) gives the topn most similar words 
model.similarity(w1 = "word 1", w2 = "word 2") returns the cosine similarity between the two words 

'''

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec("combined_embeddings.txt", "combined_embeddings_formatted.txt")
model = KeyedVectors.load_word2vec_format("combined_embeddings_formatted.txt", binary=False)


from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.translate import Alignment, AlignedSent, IBMModel1
import pickle
import dill
from math import log

def translate(vi_text):
    
    def beam_search_decoder(data, k):
        sequences = [[[], 0.0]]
        # walk over each step in sequence
        for word in data:
            all_candidates = []
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                if word not in vocab.keys():
                    candidate = [seq + [word+'_UNK'], score]
                    all_candidates.append(candidate)
                    continue
                
                for j in vocab[word].keys():
                    candidate = [seq + [j], score - log(vocab[word][j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences
    
    with open('smt/rtranstable.pk', 'rb') as f: 
        vocab = pickle.load(f)
    vi_text = vi_text.lower().split()
    result = beam_search_decoder(vi_text, 3)
    
    for seq in result:
        print(' '.join(seq[0]), "-- entropy: ", seq[1])
    
s = "tôi thích bạn , bạn có thích tôi không ?"
translate(s)


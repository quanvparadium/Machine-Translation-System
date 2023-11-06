# Vietnamese to English Statistical Machine Translation
# Dataset: EVBCorpus
# Model: IBM1 with EM algorithm

from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.translate import Alignment, AlignedSent, IBMModel1
import pickle
import dill
import math

# Path to aligned corpus dataset
FILE_PATH = 'smt/EVBCorpus_EVBNews_v2.0/'

def count_samples(soup):
    samples = 0
    text_list=soup.find_all('text')
    for item in text_list:
        lines_in_item=item.text.split('\n')
        for x in lines_in_item:
            if x.strip()!="":
                samples += 1
    return samples // 3

def get_target_sentence(soup, id, language='en'):
    sentence = soup.find_all("s", id=language + str(id))[0]
    return sentence.text.strip()

def get_source_sentence(soup, id, language='vn'):
    sentence = soup.find_all("s", id=language + str(id))[0]
    return sentence.text.strip()

def get_alignment_pairs(soup, id):
    alignments = soup.find_all("a", id="ev" + str(id))[0]
    alignments = alignments.text[:-1].split(';')
    align_pairs = []
    for align in alignments:
        tgt_idxs, src_idxs = align.split('-')
        tgt_idxs = tgt_idxs.split(',')
        src_idxs = src_idxs.split(',')
        
        align_pairs += [(int(i)-1, int(j)-1) for i in tgt_idxs for j in src_idxs]
        
    return align_pairs

def get_bitext(from_file, to_file=None, file_path = FILE_PATH):
    bitext = []
    if to_file == None:
        to_file = from_file
    for file_num in range(from_file, to_file + 1):
        file_name = ''
        if 1 <= file_num <= 9:
            file_name = 'N000' + str(file_num) + file_name
        elif 10 <= file_num <= 99:
            file_name = 'N00' + str(file_num) + file_name
        elif 100 <= file_num <= 999:
            file_name = 'N0' + str(file_num) + file_name
        else:
            file_name = 'N1000'
        file_name = file_path + file_name + '.sgml'
        
        with open(file_name, 'r', encoding="utf-8") as fo:
            sgml = fo.read()
        soup = BeautifulSoup(sgml,'html.parser')
        
        samples = count_samples(soup)
        
        for i in range(1, samples + 1):
            try:
                target_sentence = get_target_sentence(soup, i)
                source_sentence = get_source_sentence(soup, i)
                align_pairs = get_alignment_pairs(soup, i)

                bitext.append(AlignedSent(
                    target_sentence.lower().split(' '),
                    source_sentence.lower().split(' '),
                    Alignment(align_pairs)
                ))
            except:
                pass
    return bitext


class V2E_IBMModel():
    
    def __init__(self, bitext, iterations):
        self.bitext = bitext
        self.model = IBMModel1(self.bitext, iterations)
        self.trans_table = self.model.translation_table
        # self.rtrans_table = self.get_reverse_trans_table()
    
    def get_trans_table(self):
        '''
        Translation_table[t][s] is the probability that word ``t`` in the target sentence is aligned to
        word ``s`` in the source sentence.
        '''
        return self.model.translation_table         
            
    def get_reverse_trans_table(self):
        t = self.model.translation_table
        vocab = defaultdict(dict)
        en_word = None
        en_dict = None
        for en_item in t.items():
            en_word = en_item[0]
            en_dict = en_item[1]
            for vi_item in en_dict.items():
                vi_word, prob = vi_item
                vocab[vi_word][en_word] = prob
        return vocab
    
    # def save_reverse_trans_table(self, file_name='rtranstable.pk'):
    #     with open(file_name, 'wb') as f:
    #         dill.dump(self.model, f)

            

if __name__ == '__main__':
    print("Loading aligned corpus...")
    bitext = get_bitext(1, 10, FILE_PATH)
    
    print("Training IBM model...")
    ibm = V2E_IBMModel(bitext, 25)
    
    print("Reversing model's translation table...")
    rtranstable = ibm.get_reverse_trans_table()
    
    with open('smt/rtranstable.pk', 'wb') as f:
        pickle.dump(rtranstable, f)
        
    print("Reverse translation table saved.")
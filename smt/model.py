# Vietnamese to English Statistical Machine Translation
# Dataset: EVBCorpus
# Model: IBM1 with EM algorithm

from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.translate import Alignment, AlignedSent, IBMModel1
import pickle
import math

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


def get_bitext(from_file, to_file=None):
    file_path = 'EVBCorpus_EVBNews_v2.0/'
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


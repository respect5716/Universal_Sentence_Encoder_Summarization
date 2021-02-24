import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from nltk.tokenize import sent_tokenize

from utils import load_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization")
args = parser.parse_args()

def check_dir(base_dir):
    embedding_dir = os.path.join(base_dir, 'data/embedding')
    if not os.path.isdir(embedding_dir):
        os.makedirs(embedding_dir)

def get_file_list(base_dir):
    data_dir = os.path.join(base_dir, 'data/BBC News Summary')
    assert os.path.isdir(data_dir)
    all_source_files = glob(data_dir + "/News Articles/*/*.txt")
    all_target_files = [i.replace("News Articles", "Summaries") for i in all_source_files]
    return all_source_files, all_target_files

def split_data(all_source_files, all_target_files, num_test=100):
    random_idx = np.random.permutation(len(all_source_files))
    train_idx = random_idx[num_test:]
    test_idx = random_idx[:num_test]

    train_source_files = [all_source_files[i] for i in train_idx]
    train_target_files = [all_target_files[i] for i in train_idx]
    test_source_files = [all_source_files[i] for i in test_idx]
    test_target_files = [all_target_files[i] for i in test_idx]

    return train_source_files, train_target_files, test_source_files, test_target_files


def load_text(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def analyze(source_file, target_file, model):
    parse = source_file.split('\\')
    category = parse[-2]
    index = parse[-1].split('.')[0]
    name =  f'{category}_{index}'
    
    source_text = load_text(source_file)
    target_text = load_text(target_file)
    
    source_text = source_text.split('\n')
    title = source_text[0]
    sentence = []
    for i in source_text[1:]:
        sentence += sent_tokenize(i)
    
    label = [i in target_text for i in sentence]
    label = [str(int(i)) for i in label]
    label = ''.join(label)
    
    with tf.device('/cpu:0'):
        embedding = model(sentence).numpy()
    sentence = '///'.join(sentence)
    data = {'name':name, 'title':title, 'sentence':sentence, 'label':label} 
    return data, embedding

def prepare(source_files, target_files, base_dir, model):
    data = []
    for s, t in zip(source_files, target_files):
        try:
            _data, _embedding = analyze(s, t, model)
            data.append(_data)
            np.save(os.path.join(base_dir, f"data/embedding/{_data['name']}.npy"), _embedding)
        except:
            print("Error in : ", s)
    data = pd.DataFrame(data)
    return data

def main(args):
    check_dir(args.base_dir)
    model = load_model(args.base_dir)
    
    all_source_files, all_target_files = get_file_list(args.base_dir)
    train_source_files, train_target_files, test_source_files, test_target_files = split_data(all_source_files, all_target_files)
    
    train_data = prepare(train_source_files, train_target_files, args.base_dir, model)
    test_data = prepare(test_source_files, test_target_files, args.base_dir, model)
    
    train_data.to_csv(os.path.join(args.base_dir, 'data/train_data.csv'), index=False)
    test_data.to_csv(os.path.join(args.base_dir, 'data/test_data.csv'), index=False)


if __name__ == '__main__':
    main(args)
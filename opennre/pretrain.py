import requests

from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np
import logging
import typing as t

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
default_root_path = os.path.join(os.getenv('USERPROFILE'), '.opennre')


def download_wiki80(root_path=default_root_path):
    _download_files_from_extensions(root_path,
                                    ['/benchmark/wiki80/wiki80_rel2id.json',
                                     '/benchmark/wiki80/wiki80_val.txt',
                                     '/benchmark/wiki80/wiki80_train.txt'])


def download_tacred(root_path=default_root_path):
    _download_file_from_extension(root_path, '/benchmark/tacred/tacred_rel2id.json')
    logging.info(
        'Due to copyright limits, we only provide rel2id for TACRED. '
        'Please download TACRED manually and convert the data to OpenNRE format if needed.')


def download_nyt10(root_path=default_root_path):
    _download_file_from_extension(root_path,
                                  ['/benchmark/nyt10/nyt10_rel2id.json',
                                   '/benchmark/nyt10/nyt10_train.txt',
                                   '/benchmark/nyt10/nyt10_test.txt',
                                   '/benchmark/nyt10/nyt10_val.txt'])


def download_wiki_distant(root_path=default_root_path):
    _download_files_from_extensions(root_path,
                                    ['/benchmark/wiki_distant/wiki_distant_rel2id.json',
                                     '/benchmark/wiki_distant/wiki_distant_train.txt',
                                     '/benchmark/wiki_distant/wiki_distant_test.txt',
                                     '/benchmark/wiki_distant/wiki_distant_val.txt'])


def download_semeval(root_path=default_root_path):
    _download_files_from_extensions(root_path,
                                    ['/benchmark/semeval/semeval_rel2id.json',
                                     '/benchmark/semeval/semeval_train.txt',
                                     '/benchmark/semeval/semeval_test.txt',
                                     '/benchmark/semeval/semeval_val.txt'])


def download_glove(root_path=default_root_path):
    _download_files_from_extensions(root_path,
                                    ['/pretrain/glove/glove.6B.50d_mat.npy',
                                     '/pretrain/glove/glove.6B.50d_word2id.json'])


def download_bert_base_uncased(root_path=default_root_path):
    _download_files_from_extensions(root_path,
                                    ['/pretrain/bert-base-uncased/config.json',
                                     '/pretrain/bert-base-uncased/pytorch_model.bin',
                                     '/bert-base-uncased/vocab.txt'])


def _download_files_from_extensions(root_path: str, path_extensions: t.List[str]):
    for extension in path_extensions:
        _download_file_from_extension(root_path, extension)


def _download_file_from_extension(root_path: str, path_extension: str):
    url = f'{root_url}/opennre{path_extension}'
    path = f'{root_path}{path_extension}'
    _download_file_if_not_exists(url, path)


def _download_file_if_not_exists(url: str, path: str):
    if os.path.exists(path):
        return
    print(f"Downloading {url}")
    request = requests.get(url, allow_redirects=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb+').write(request.content)
    print(f"Downloaded {url}")


def download_pretrain(model_name, root_path=default_root_path):
    path_extension = f'/pretrain/nre/{model_name}.pth.tar'
    _download_file_from_extension(root_path, path_extension)


def _safe_get(url: str, path: str):
    _download_file_if_not_exists(url, path)
    return open(path)


def get_model(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if model_name == 'wiki80_cnn_softmax':
        download_pretrain(model_name, root_path=root_path)
        download_glove(root_path=root_path)
        download_wiki80(root_path=root_path)
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(
            token2id=wordi2d,
            max_length=40,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['wiki80_bert_softmax', 'wiki80_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download_bert_base_uncased(root_path=root_path)
        download_wiki80(root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['tacred_bert_softmax', 'tacred_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download_bert_base_uncased(root_path=root_path)
        download_tacred(root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/tacred/tacred_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError

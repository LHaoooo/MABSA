# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from squad.squad_evaluate import exact_match_score
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, \
    RawFinalResult, wrapped_get_final_text, id_to_label

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def metric_max_over_ground_truths(metric_fn, term, polarity, gold_terms, gold_polarities):
    hit = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        score = metric_fn(term, gold_term)
        if score and polarity == gold_polarity:
            hit = 1
    return hit

def mate_metric(metric_fn, term, gold_terms):
    goodnum = 0
    for gold_term in gold_terms:
        score = metric_fn(term, gold_term)
        if score:
            goodnum = 1
    return goodnum

def masc_metric(metric_fn, term, polarity, gold_terms, gold_polarities):
    classygood = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        if metric_fn(term, gold_term) and polarity == gold_polarity :
            classygood = 1
    return classygood

def eval_absa(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                # print(final_text)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred_terms': pred_terms, 'pred_polarities': pred_polarities,'gold_terms':example.term_texts,'gold_polarites':example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in dict(zip(pred_terms, pred_polarities)).items():
            # print("term:",term)
            # print("polarity:",polarity)
            # print('all terms:',example.term_texts)
            # print("all polarities:",example.polarities)
            common+= metric_max_over_ground_truths(exact_match_score, term, polarity, example.term_texts, example.polarities)
            
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def eval_mate(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        
        for start_index, end_index, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)

        prediction = {'pred_terms': pred_terms, 'gold_terms':example.term_texts}
        all_nbest_json[example.example_id] = prediction

        for term in pred_terms:
            # print("term:",term)
            # print('all terms:',example.term_texts)
            common += mate_metric(exact_match_score, term, example.term_texts)
            
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'mate_p': p, 'mate_r': r, 'mate_f1': f1, 'mate_common': common, 'mate_retrieved': retrieved, 'mate_relevant': relevant}, all_nbest_json

def eval_masc(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    # enti= 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        input_terms = []
        pred_polarities = []
        
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):

            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                input_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'input_terms': input_terms, 'pred_polarities': pred_polarities,'gold_terms':example.term_texts,'gold_polarites':example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(input_terms, pred_polarities):
            # print("term:",term)
            # print('all terms:',example.term_texts)
            # print("polarity:",polarity)
            # print("all polarities:",example.polarities)
            common += masc_metric(exact_match_score, term, polarity, example.term_texts, example.polarities)
            
        retrieved += len(pred_polarities)
        relevant += len(example.polarities)

    acc = common / relevant if relevant > 0 else 0.

    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant if relevant > 0 else 0.
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'masc_acc': acc, 'masc_f1': f1, 'masc_common': common, 'masc_retrieved': retrieved, 'masc_relevant': relevant}, all_nbest_json

if __name__=='__main__':
    eval_absa()

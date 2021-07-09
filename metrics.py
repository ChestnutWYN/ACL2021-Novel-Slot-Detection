from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.span_based_f1_measure import TAGS_TO_SPANS_FUNCTION_TYPE
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import copy
import math

class NSDSpanBasedF1Measure():
    def __init__(
        self,
        tag_namespace: str = "tags",
        ignore_classes: List[str] = None,
        label_encoding: Optional[str] = "BIO", 
        tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None, 
        nsd_slots: List[str] = None) -> None:
        """
        SpanBasedF1Measure
        - IND, for f1-ind|precision-ind|recall-ind
        - NSD, for f1-nsd|precision-nsd|recall-nsd
        """
        self._ignore_classes: List[str] = ignore_classes or []
        self._nsd_slots = nsd_slots or ["ns"]

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
    def __call__(
        self,
        gold_labels: pd.Series ,
        predict_labels: pd.Series,
        rose: bool = False,
        proportion:float = 1,
    ):
        """
        Params:
            - gold_labels: A tensor of predictions of shape (batch_size, sequence_length)
            - predict_labels: A tensor of predictions of shape (batch_size, sequence_length)
            - rose：Bool，Whether use the new metric —— restriction-oriented span evaluation (ROSE)
            - proportion, The parameter of ROSE, We consider a span is correct when the number of correctly predicted tokens is greater than the proportion p of the span length. 

        Returns:
            - all_metrics: dict

        ROSE:
            To meet a reasonable NSD scenario, we propose a new metric, restriction-oriented span evaluation(ROSE).
            In contrast to traditional spanF1, consider the following two conditions:
                Condition 1: When the tokens prediction exceeds the span.
                Condition 2: When the number of correctly predicted tokens is greater than a settable proportion p of the span length. 

        """
        batch_size = gold_labels.size
        for i in range(batch_size):
            gold_spans = bio_tags_to_spans(gold_labels[i],self._ignore_classes)
            predict_spans = bio_tags_to_spans(predict_labels[i],self._ignore_classes)
            if rose:
                gold_ns_spans = [span for label,span in gold_spans if label=="ns"]
                predict_ns_spans = [span for label,span in predict_spans if label=="ns"]
                
                need_remove_span = []
                _gold_ns_spans = copy.deepcopy(gold_ns_spans)
                for span in gold_ns_spans:
                    gold_span_start, gold_span_end = span
                    tolerance = math.floor(proportion * (gold_span_end - gold_span_start + 1)) if (proportion * (gold_span_end - gold_span_start + 1)) > 1 else 1# gold span length中必须预测>= tolerance才可以
                    great_num = 0
                    gold_idx = np.linspace(gold_span_start,gold_span_end,gold_span_end-gold_span_start+1,dtype=int).tolist()
                    need_merge_span = []
                    tp_flag = True
                    for pre_span in predict_ns_spans:
                        pre_span_start,pre_span_end = pre_span
                        # condition1
                        if pre_span_start <= gold_span_start and pre_span_end >= gold_span_end:
                            self._true_positives["ns"] += 1
                            _gold_ns_spans.remove(span)
                            need_remove_span += [pre_span]
                        # condition 2
                        elif pre_span_start <= gold_span_end and pre_span_end >= gold_span_start:
                            need_merge_span.append(pre_span)
                            predict_idx = np.linspace(pre_span_start,pre_span_end,pre_span_end-pre_span_start+1,dtype=int).tolist()
                            great_num += len(list(set(gold_idx).intersection(predict_idx)))
                            if great_num >= tolerance and tp_flag == True:
                                self._true_positives["ns"] += 1
                                _gold_ns_spans.remove(span)
                                need_remove_span += need_merge_span
                                need_merge_span = []
                                tp_flag = False
                            elif great_num > tolerance and tp_flag == False:
                                need_remove_span += need_merge_span

                _predict_ns_spans = list(set(predict_ns_spans)-set(need_remove_span))

                self._false_positives["ns"] += len(_predict_ns_spans)
                self._false_negatives["ns"] += len(_gold_ns_spans)



            else:
                for span in predict_spans:
                    if span in gold_spans:
                        self._true_positives[span[0]] += 1
                        gold_spans.remove(span)
                    else:
                        self._false_positives[span[0]] += 1
                # These spans weren't predicted.
                for span in gold_spans:
                    self._false_negatives[span[0]] += 1
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.

        (*) Additionally, for novel slots and normal (not novel)
        slots, an ``ns`` key and a ``normal`` key are included respectively, which
        provide the precision, recall and f1-measure for all novel spans
        and all normal spans, respectively.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                            self._false_positives[tag],
                                            self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-overall"] = f1_measure
        # all_metrics["acc-overall"] = sum(self._true_positives.values()/sum()
        


        # (*) Compute the precision, recall and f1 for all nsd spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._true_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._false_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._false_negatives.items()))))
        all_metrics["precision-nsd"] = precision
        all_metrics["recall-nsd"] = recall
        all_metrics["f1-nsd"] = f1_measure

        # (*) Compute the precision, recall and f1 for all ind spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._true_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._false_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._false_negatives.items()))))
        all_metrics["precision-ind"] = precision
        all_metrics["recall-ind"] = recall
        all_metrics["f1-ind"] = f1_measure

        if reset:
            self.reset()
        return all_metrics
    def _compute_metrics(self,true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)


import numpy as np
import pandas as pd
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
from sklearn.metrics import confusion_matrix,classification_report
from metrics import NSDSpanBasedF1Measure


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray: 
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)

    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result

def parse_line(line:List): # [for Spanf1]
    """
    Given the predicted sequence (contain "ns"), return the parsed BIO sequence (contain "B-ns" and "I-ns"). 
    And due to the override, IND tags may be discordant, so we need to adjust the IND tags after been override.
    e.g.
        Input : ns ns ns O B-playlist_owner B-playlist ns I-playlist O 
        Return : B-ns I-ns I-ns O B-playlist_owner B-playlist B-ns B-playlist O 

    Params:
        - line(list),the prediced sequence (contain "ns"). 

    Returns:
        - adjust_line(list), the parsed BIO sequence(contain "B-ns" and "I-ns")
    """
    adjust_line = []
    for i,label in enumerate(line):
        if label in ["ns","B-ns","I-ns"]:
            if i == 0:
                adjust_line.append("B-ns")
            elif adjust_line[i-1] in ["ns","B-ns","I-ns"]:
                adjust_line.append("I-ns")
            else:
                adjust_line.append("B-ns")
        elif label == "O":
            adjust_line.append("O")
        else:
            if i == 0:
                adjust_line.append(label)
            elif adjust_line[i-1][-3:] == "-ns" and label[:2]=="I-":
                adjust_line.append("B-"+label[-2:])
            else:
                adjust_line.append(label)
    return adjust_line

def parse_token(line:List): # [for tokenf1]
    """
    Given the predicted sequence (contain "ns" or "B-ns", "I-ns"), return the parsed BIO sequence (only contain "B-ns"). 
    e.g.
        Input : ns ns ns O B-playlist_owner B-playlist I-playlist O 
        Return : B-ns B-ns B-ns O B-playlist_owner B-playlist I-playlist O 

    Params:
        - line(list),the prediced sequence (contain "ns" or "B-ns", "I-ns"). 

    Returns:
        - adjust_line(list), the parsed BIO sequence (contain "B-ns")
    """
    adjust_line = []
    for i,label in enumerate(line):
        if label in ["ns","B-ns","I-ns"]:
            adjust_line.append("B-ns")
        elif label == "O":
            adjust_line.append("O")
        else:
            if i == 0:
                adjust_line.append(label)
            elif line[i-1][-2:] == "ns" and label[:2]=="I-":
                adjust_line.append("B-"+label[-2:])
            else:
                adjust_line.append(label)
    return adjust_line


def token_metric(true:list,predict:list):
    """
    Get token-level metrics.
    """
    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                        )

    spanf1(pd.Series([true]),pd.Series([predict]),False)
    token_metrics = spanf1.get_metric(reset=True)
    f_nsd = token_metrics["f1-nsd"]
    r_nsd = token_metrics["recall-nsd"]
    p_nsd = token_metrics["precision-nsd"]

    print(f"=====> Token(Experiment) <=====")
    print(f"NSD:  f:{f_nsd}, p:{p_nsd}, r:{r_nsd}\n")

def rose_metric(test_true_lines:list,test_pred_lines:list):
    """
    To meet a reasonable NSD scenario, we propose a new metric, restriction-oriented span evaluation(ROSE).
    We consider a span is correct:
        - When the tokens prediction exceeds the span.
        - When the number of correctly predicted tokens is greater than a settable proportion p of the span length. 

    Params:
        - test_true_lines, (seq_dim,token_dim) contation BIO tags.
        - test_pred_lines, (seq_dim,token_dim) contation BIO tags.

    Returns:
        - 
    """
    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                    )
    
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.25)
    nsd_metrics_25 = spanf1.get_metric(reset=True)
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.5)
    nsd_metrics_50 = spanf1.get_metric(reset=True)
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.75)
    nsd_metrics_75 = spanf1.get_metric(reset=True)
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,1)
    nsd_metrics_100 = spanf1.get_metric(reset=True)

    rose_f1_25 = nsd_metrics_25["f1-nsd"].round(2)
    rose_f1_50 = nsd_metrics_50["f1-nsd"].round(2)
    rose_f1_75 = nsd_metrics_75["f1-nsd"].round(2)
    rose_f1_100 = nsd_metrics_100["f1-nsd"].round(2)
    rose_p_25 = nsd_metrics_25["precision-nsd"].round(2)
    rose_p_50 = nsd_metrics_50["precision-nsd"].round(2)
    rose_p_75 = nsd_metrics_75["precision-nsd"].round(2)
    rose_p_100 = nsd_metrics_100["precision-nsd"].round(2)
    rose_r_25 = nsd_metrics_25["recall-nsd"].round(2)
    rose_r_50 = nsd_metrics_50["recall-nsd"].round(2)
    rose_r_75 = nsd_metrics_75["recall-nsd"].round(2)
    rose_r_100 = nsd_metrics_100["recall-nsd"].round(2)

    print(f"=====> ROSE(Experiment) <=====")
    print(f"ROSE-25%:  f:{rose_f1_25}, p:{rose_p_25}, r:{rose_r_25}")
    print(f"ROSE-50%:  f:{rose_f1_50}, p:{rose_p_50}, r:{rose_r_50}")
    print(f"ROSE-75%:  f:{rose_f1_75}, p:{rose_p_75}, r:{rose_r_75}")
    print(f"ROSE-100%:  f:{rose_f1_100}, p:{rose_p_100}, r:{rose_r_100}")

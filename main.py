from allennlp.models import model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data import DataIterator#
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from allennlp.common.util import prepare_global_logging, cleanup_global_logging, prepare_environment
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data import vocabulary
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models import NSDSlotTaggingModel
from predictors import SlotFillingPredictor
from dataset_readers import MultiFileDatasetReader
from metrics import NSDSpanBasedF1Measure
from utils import *

from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
from time import *
import numpy as np
import pandas as pd
import argparse
import os
import logging

vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode",type=str,choices=["train", "test", "both"], default="test",
                        help="Specify running mode: only train, only test or both.")
    arg_parser.add_argument("--dataset",type=str,choices=["SnipsNSD5%", "SnipsNSD15%", "SnipsNSD30%"], default=None,
                        help="The dataset to use.")
    arg_parser.add_argument("--output_dir",type=str, default="./output",
                        help="The path of trained model.")
    arg_parser.add_argument("--cuda",type=int, default=1,
                        help="cuda device.")
    arg_parser.add_argument("--threshold", type=float,default=None,
                        help="The specified threshold value.")
    arg_parser.add_argument("--batch_size",type=int, default=200,
                        help="Batch size.")
    args = arg_parser.parse_args()
    return args

def SlotTrain(config_path,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    params = Params.from_file(config_path)
    stdout_handler = prepare_global_logging(output_dir, False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(file_path=params.pop("train_data_path", None))
    valid_dataset = reader.read(params.pop("validation_data_path", None))
    test_data_path = params.pop("test_data_path", None)
    if test_data_path:
        test_dataset = reader.read(test_data_path)
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    else:
        test_dataset = None
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=vocab)
    vocab.save_to_files(os.path.join(output_dir, "vocabulary"))
    # copy config file
    with open(config_path, "r", encoding="utf-8") as f_in:
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())

    iterator = DataIterator.from_params(params.pop("iterator", None))
    iterator.index_with(vocab)
    
    trainer_params = params.pop("trainer", None)
    trainer = Trainer.from_params(model=model,
                                serialization_dir=output_dir,
                                iterator=iterator,
                                train_data=train_dataset,
                                validation_data=valid_dataset,
                                params=trainer_params.duplicate())
    trainer.train()

    # evaluate on the test set
    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch  # import here to ensure the republication of the experiment
        model.load_state_dict(torch.load(os.path.join(output_dir, "best.th")))
        test_metrics = evaluate(model, test_dataset, iterator,
                                cuda_device=trainer_params.pop("cuda_device", 1),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")
    cleanup_global_logging(stdout_handler)

args = parse_args()

# Train
if args.mode in ["train","both"]:

    output_dir = os.path.join(args.output_dir,args.dataset)
    config_path = "./config/"+args.dataset+".json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    params = Params.from_file(config_path)
    stdout_handler = prepare_global_logging(output_dir, False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(file_path=params.pop("train_data_path", None))
    valid_dataset = reader.read(params.pop("validation_data_path", None))
    test_data_path = params.pop("test_data_path", None)
    if test_data_path:
        test_dataset = reader.read(test_data_path)
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    else:
        test_dataset = None
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=vocab)
    vocab.save_to_files(os.path.join(output_dir, "vocabulary"))
    # copy config file
    with open(config_path, "r", encoding="utf-8") as f_in:
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())

    iterator = DataIterator.from_params(params.pop("iterator", None))
    iterator.index_with(vocab)
    
    trainer_params = params.pop("trainer", None)
    trainer = Trainer.from_params(model=model,
                                serialization_dir=output_dir,
                                iterator=iterator,
                                train_data=train_dataset,
                                validation_data=valid_dataset,
                                params=trainer_params.duplicate())
    trainer.train()

    # evaluate on the test set
    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch  # import here to ensure the republication of the experiment
        model.load_state_dict(torch.load(os.path.join(output_dir, "best.th")))
        test_metrics = evaluate(model, test_dataset, iterator,
                                cuda_device=trainer_params.pop("cuda_device", 1),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")
    cleanup_global_logging(stdout_handler)


# Test
if args.mode in ["test","both"]:
    if args.mode == "both":
        model_dir = output_dir
    else:
        model_dir = os.path.join(args.output_dir,args.dataset)
    # predict
    archive = load_archive(model_dir,cuda_device=args.cuda)
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_filling_predictor")
    train_outputs = predictor.predict_multi(file_path = os.path.join("data",args.dataset,"train") ,batch_size = args.batch_size)
    test_outputs = predictor.predict_multi(file_path = os.path.join("data",args.dataset,"test") ,batch_size = args.batch_size)
    ns_labels = ["ns","B-ns","I-ns"]

    # GDA
    gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
    gda.fit(np.array(train_outputs["encoder_outs"]), train_outputs["true_labels"])
    gda_means = gda.means_ 

    test_gda_result = confidence(np.array(test_outputs["encoder_outs"]), gda.means_, "euclidean", gda.covariance_)
    test_score = pd.Series(test_gda_result.min(axis=1))
    test_ns_idx = [idx_vo for idx_vo , _vo in enumerate(test_outputs["true_labels"]) if _vo in ns_labels]
    test_ind_idx = [idx_vi for idx_vi , _vi in enumerate(test_outputs["true_labels"]) if _vi not in ns_labels]
    test_ns_score = test_score[test_ns_idx]
    test_ind_score = test_score[test_ind_idx]

    # threshold
    threshold = args.threshold
    
    # override
    test_y_ns = pd.Series(test_outputs["predict_labels"])
    test_y_ns[test_score[test_score> threshold].index] = "ns"
    test_y_ns = list(test_y_ns)

    # Metrics —— ROSE
    start_idx = 0
    end_idx = 0
    test_pred_lines = []
    test_true_lines = []
    seq_lines = pd.DataFrame(test_outputs["tokens"])
    for i,seq in enumerate(seq_lines["tokens"]):
        start_idx = end_idx
        end_idx = start_idx + len(seq)
        adju_pred_line = parse_line(test_y_ns[start_idx:end_idx])
        test_true_line = test_outputs["true_labels"][start_idx:end_idx]
        test_pred_lines.append(adju_pred_line)
        test_true_lines.append(test_true_line)
    rose_metric(test_true_lines,test_pred_lines)

    # Metrics —— Token
    test_pred_tokens = parse_token(test_y_ns)
    test_true_tokens = parse_token(test_outputs["true_labels"])
    token_metric(test_true_tokens,test_pred_tokens)

    


    

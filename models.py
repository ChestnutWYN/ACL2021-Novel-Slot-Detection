
from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from pytorch_pretrained_bert import BertTokenizer

from typing import Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)

@Model.register("bert_st")
class NSDSlotTaggingModel(Model):
    
    def __init__(self, 
                 vocab: Vocabulary,
                 bert_embedder: Optional[PretrainedBertEmbedder] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: Optional[float] = None,
                 use_crf: bool = True) -> None:
        super().__init__(vocab)

        if bert_embedder:
            self.use_bert = True
            self.bert_embedder = bert_embedder
        else:
            self.use_bert = False
            self.basic_embedder = BasicTextFieldEmbedder({
                "tokens": Embedding(vocab.get_vocab_size(namespace="tokens"), 1024)
            })
            self.rnn = Seq2SeqEncoder.from_params(Params({     
                "type": "lstm",
                "input_size": 1024,
                "hidden_size": 512,
                "bidirectional": True,
                "batch_first": True
            }))

        self.encoder = encoder

        if encoder:
            hidden2tag_in_dim = encoder.get_output_dim()
        else:
            hidden2tag_in_dim = bert_embedder.get_output_dim()
        self.hidden2tag = TimeDistributed(torch.nn.Linear(
            in_features=hidden2tag_in_dim,
            out_features=vocab.get_vocab_size("labels")))
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.use_crf = use_crf
        if use_crf:
            crf_constraints = allowed_transitions(
                constraint_type="BIO",
                labels=vocab.get_index_to_token_vocabulary("labels")
            )
            self.crf = ConditionalRandomField(
                num_tags=vocab.get_vocab_size("labels"),
                constraints=crf_constraints,
                include_start_end_transitions=True
            )
        
        self.f1 = SpanBasedF1Measure(vocab, 
                                     tag_namespace="labels",
                                     ignore_classes= None ,
                                     label_encoding="BIO")

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                wordnet: Dict[str, torch.Tensor] = None,
                slot_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Return a Dict (str -> torch.Tensor), which contains fields:
            mask - the mask matrix of ``sentence``, shape: (batch_size, seq_length)
            embeddings - the embedded tokens, shape: (batch_size, seq_length, embed_size)
            encoder_out - the output of contextual encoder, shape: (batch_size, seq_length, num_features)
            tag_logits - the output of tag projection layer, shape: (batch_size, seq_length, num_tags)
            predicted_tags - the output of CRF layer (use viterbi algorithm to obtain best paths),
                             shape: (batch_size, seq_length)
        """

        output = {}

        mask = get_text_field_mask(sentence)
        output["mask"] = mask
        
        if self.use_bert:
            embeddings = self.bert_embedder(sentence["bert"], sentence["bert-offsets"], sentence["bert-type-ids"])
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
        else:
            embeddings = self.basic_embedder(sentence)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
            embeddings = self.rnn(embeddings, mask)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["rnn_out"] = embeddings
        
        if self.encoder:
            encoder_out = self.encoder(embeddings, mask)
            if self.dropout:
                encoder_out = self.dropout(encoder_out)
            output["encoder_out"] = encoder_out
        else:
            encoder_out = embeddings
        
        tag_logits = self.hidden2tag(encoder_out)
        output["tag_logits"] = tag_logits

        if self.use_crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            predicted_tags = [x for x, y in best_paths]  # get the tags and ignore the score
            output["predicted_tags"] = predicted_tags
        else:
            output["predicted_tags"] = torch.argmax(tag_logits, dim=-1)  # pylint: disable=no-member
        
        if slot_labels is not None:
            if self.use_crf:
                log_likelihood = self.crf(tag_logits, slot_labels, mask)  # returns log-likelihood
                output["loss"] = -1.0 * log_likelihood  # add negative log-likelihood as loss
                
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = tag_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
                self.f1(class_probabilities, slot_labels, mask.float())
            else:
                output["loss"] = sequence_cross_entropy_with_logits(tag_logits, slot_labels, mask)
                self.f1(tag_logits, slot_labels, mask.float())
        
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        matric = self.f1.get_metric(reset)
        return {"precision": matric["precision-overall"],
                "recall": matric["recall-overall"],
                "f1": matric["f1-measure-overall"]}
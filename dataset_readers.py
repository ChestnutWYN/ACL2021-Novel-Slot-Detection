

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary

from typing import Optional, Iterator, List, Dict
import os

@DatasetReader.register("multi_file")
class MultiFileDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    
    def text_to_instance(self, tokens: List[str], slots: List[str] = None) -> Instance:
        sentence_field = TextField([Token(token) for token in tokens], self.token_indexers)
        fields = {"sentence": sentence_field}
        if slots:
            slot_label_field = SequenceLabelField(labels=slots, sequence_field=sentence_field)
            fields["slot_labels"] = slot_label_field
        
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        token_file_path = os.path.join(file_path, "seq.in")
        label_file_path = os.path.join(file_path, "seq.out")
        with open(token_file_path, "r", encoding="utf-8") as f_token:
            token_lines = f_token.readlines()
        with open(label_file_path, "r", encoding="utf-8") as f_label:
            label_lines = f_label.readlines()
        assert len(token_lines) == len(label_lines)
        for token_line, label_line in zip(token_lines, label_lines):
            if not token_line.strip() or not label_line.strip():
                continue
            tokens: List[str] = token_line.strip().split(" ")
            labels: List[str] = label_line.strip().split(" ")
            if len(tokens) == 0 or len(labels) == 0:
                continue
            tokens = [token.strip() for token in tokens if token.strip()]
            labels = [label.strip() for label in labels if label.strip()]
            assert len(tokens) == len(labels)
            yield self.text_to_instance(tokens, labels)

from typing import Dict
import json
import logging
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("pc_tweets")
class PriorityCrisisDatasetReader(DatasetReader):
    """
       Reads a JSON-lines file containing tweets with priority levels and creates a
       dataset suitable for priority classification using these tweets.
       Expected format for each input line: {"text": "text", "label": "text"}
       The JSON could have other fields, too, but they are ignored.
       The output of ``read`` is a list of ``Instance`` s with the fields:
           text: ``TextField``
           label: ``LabelField``
       where the ``label`` is derived from the label of the tweet.
       Parameters
       ----------
       lazy : ``bool`` (optional, default=False)
           Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
           take longer per batch.  This also allows training with datasets that are too large to fit
           in memory.
       tokenizer : ``Tokenizer``, optional
           Tokenizer to use to split the tweet's text into words or other kinds of tokens.
           Defaults to ``WordTokenizer()``.
       token_indexers : ``Dict[str, TokenIndexer]``, optional
           Indexers used to define input token representations. Defaults to ``{"tokens":
           SingleIdTokenIndexer()}``.
       """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        # self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                text = paper_json['text']
                label = paper_json['priority']
                if label == "<NULL>":
                    continue
                yield self.text_to_instance(text, label)

    @overrides
    def text_to_instance(self, text: str, label: str=None) -> Instance:
        # tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text = [Token(token) for token in text.split()]
        tokenized_text = TextField(tokenized_text, self._token_indexers)
        fields = {'text': tokenized_text}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

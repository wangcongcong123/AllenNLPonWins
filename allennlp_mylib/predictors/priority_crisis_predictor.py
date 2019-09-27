from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('priority_crisis_predictor')
class PriorityCrisisClassifierPredictor(Predictor):
    """Predictor wrapper for the PriorityCrisisClassifier"""

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        instance = self._json_to_instance(json_dict)
        # label_dict will be like {0: "Low", 1: "Medium", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["Low", "Medium", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        gt_label = None
        post_id = None
        event_id = None
        if "priority" in json_dict:
            gt_label = json_dict['priority']
        if "post_id" in json_dict:
            post_id = json_dict["post_id"]
        if "event_id" in json_dict:
            event_id = json_dict["event_id"]
        return {"instance": self.predict_instance(instance), "all_labels": all_labels, "gt_label": gt_label,
                "post_id": post_id, "event_id": event_id}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text=text)

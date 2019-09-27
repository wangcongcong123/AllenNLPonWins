import sys

from allennlp.commands import main

sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "tmp/my_debug",
    "dataset/dataset_fixure.json",
    "--include-package", "allennlp_mylib",
    "--predictor", "priority_crisis_predictor",
    "--output-file", "predictions/predicted-test",
    "--cuda-device","0"
]

main()
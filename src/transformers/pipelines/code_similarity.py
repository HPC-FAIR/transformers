from transformers import pipeline
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from .base import PIPELINE_INIT_ARGS
from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

logger = logging.get_logger(__name__)
@add_end_docstrings(PIPELINE_INIT_ARGS)
class CodeSimilarityPipeline(pipeline.Pipeline):
    """
    Code similarity pipeline using any 'AutoModelForSequenceClassification'. This pipeline predicts the similarity between two code snippets.
    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="codebert-base")
    >>> code1 = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
    >>> code2 = "def fib(n):\n    if n <= 1:\n        return n\n    else:\n        return fib(n-1) + fib(n-2)"
    >>> classifier(code1, code2)
    'score': 0.563736617565155
    ```
    """


    def __init__(self, model='microsoft/codebert-base', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=2)
        # self.save_pretrained('./code_similarity_pipeline')

    def _forward(self, inputs):
        # Tokenize the input codes
        encoded_text = self.tokenizer(inputs["code1"], inputs["code2"], padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Get the model's prediction for the input codes
        with torch.no_grad():
            logits = self.model(**encoded_text.to(self.device))[0]

        return logits

    def postprocess(self, text1, text2):
        inputs = {"text1": text1, "text2": text2}
        logits = self._forward(inputs)
        similarity_score = torch.softmax(logits, dim=1)[0][1].item()
        print("Score:", similarity_score)
        return similarity_score


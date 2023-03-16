from typing import Dict

from .base import GenericTensor, Pipeline
import torch

# Can't use @add_end_docstrings(PIPELINE_INIT_ARGS) here because this one does not accept `binary_output`
class CodeSimilarityPipeline(Pipeline):

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, code1:str, code2:str, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        # model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenize_kwargs)
        model_inputs1 = self.tokenizer(code1, return_tensors=return_tensors, **tokenize_kwargs)
        model_inputs2 = self.tokenizer(code2, return_tensors=return_tensors, **tokenize_kwargs)
        return model_inputs1, model_inputs2

    def _forward(self, model_inputs1, model_inputs2):
        model_outputs1 = self.model(**model_inputs1)
        model_outputs2 = self.model(**model_inputs2)

        return model_outputs1, model_outputs2

    def postprocess(self, model_outputs1, model_outputs2, return_tensors=False):
        # [0] is the first available tensor, logits or last_hidden_state.
        # if return_tensors:
        #     return model_outputs[0]
        # if self.framework == "pt":
        #     return model_outputs[0].tolist()
        # elif self.framework == "tf":
        #     return model_outputs[0].numpy().tolist()
        cosi = torch.nn.CosineSimilarity(dim=0)
        output = cosi(model_outputs1[0], model_outputs2[0])
        return output

    def __call__(self, code1, code2, *args, **kwargs):
        # return super().__call__(*args, **kwargs)
        model_inputs1, model_inputs2 = self.preprocess(code1, code2, **kwargs)
        model_outputs1, model_outputs2 = self._forward(model_inputs1, model_inputs2)
        output = self.postprocess(model_outputs1, model_outputs2, return_tensors=False)
        similarity_score = (output + 1) / 2
        print("similarity results: ", similarity_score)









# from typing import List

# from torch import Tensor
# from transformers import Pipeline


# from typing import Any, Dict, List, Tuple
# from transformers import Pipeline
# import re


# class CodeSimilarityPipeline(Pipeline):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     # _sanitize_parameters exists to allow users to pass 
#     # any parameters whenever they wish, be it at initialization 
#     # time pipeline(...., maybe_arg=4) or at call time 
#     # pipe = pipeline(...); output = pipe(...., maybe_arg=4).
#     def _sanitize_parameters(self, **kwargs):
#         preprocess_kwargs = {}
#         if "model" in kwargs:
#             preprocess_kwargs["model"] = kwargs["model"]
#         return preprocess_kwargs, {}, {}

#     def preprocess(self, inputs):
#         model_input = Tensor(inputs["input_ids"])
#         return {"model_input": model_input}
    
#     def _forward(self, model_inputs):
#         # model_inputs == {"model_input": model_input}
#         outputs = self.model(**model_inputs)
#         # Maybe {"logits": Tensor(...)}
#         return outputs
    
#     def postprocess(self, model_outputs):
#         best_class = model_outputs["logits"].softmax(-1)
#         return best_class

#         # class CodeSimilarityPipeline(Pipeline):
#         #     def __init__(self):
#         #         super().__init__(
#         #             # Define the inputs to the pipeline
#         #             # We don't need any specific model or tokenizer for this pipeline
#         #             # so we can leave these arguments as None
#         #             # model=None,
#         #             tokenizer=None,
#         #             # Define the output of the pipeline
#         #             output_type=Dict[str, float]
#         #         )

#         #     def _sanitize_parameters(self, code1: str, code2: str) -> Tuple[str, str]:
#         #         if not isinstance(code1, str) or not isinstance(code2, str):
#         #             raise ValueError("Both inputs must be strings of code.")
#         #         return code1, code2

#         #     def preprocess(self, inputs: Tuple[str, str]) -> Tuple[str, str]:
#         #         return tuple(self._remove_irrelevant_elements(code) for code in inputs)

#         #     def _remove_irrelevant_elements(self, code: str) -> str:
#         #         # Replace comments with empty string
#         #         code = re.sub(r'#.*', '', code)
#         #         # Remove white space and new lines
#         #         code = re.sub(r'\s+', '', code)
#         #         return code

#         #     def _forward(self, inputs: Tuple[str, str]) -> Dict[str, float]:
#         #         # Here you can implement the code similarity check logic
#         #         # using any suitable library or method.
#         #         # For example, you can use the difflib library:
#         #         import difflib
#         #         similarity_ratio = difflib.SequenceMatcher(None, *inputs).ratio()
#         #         return {'similarity_ratio': similarity_ratio}

#         #     def postprocess(self, outputs: Dict[str, float]) -> float:
#         #         return outputs['similarity_ratio']

#         # class CodeSimilarityPipeline(Pipeline):
#         #     def __init__(self, *args, **kwargs):
#         #         super().__init__(*args, **kwargs)

#         #     def _sanitize_parameters(self, **kwargs):
#         #         preprocess_kwargs = {}
#         #         if "input" in kwargs:
#         #             preprocess_kwargs["input"] = kwargs["input"]
#         #         return preprocess_kwargs, {}, {}

#         #     def preprocess(self, inputs, )

#         # class CodeSimilarityPipeline(Pipeline):
#         #     def __init__(self, *args, **kwargs):
#         #         super().__init__(*args, **kwargs)

#         #     # def _sanitize_parameters(self, inputs):
#         #     #     if not isinstance(inputs, list):
#         #     #         raise ValueError("inputs must be a list of two code snippets.")
#         #     #     if len(inputs) != 2:
#         #     #         raise ValueError("inputs must contain exactly two code snippets.")
#         #     #     return inputs
#         #     def _sanitize_parameters(self, **kwargs):
#         #         if 'inputs' in kwargs:
#         #             inputs = kwargs['inputs']
#         #         else:
#         #             raise ValueError("inputs must be provided as a list of two strings of code.")
#         #         if not isinstance(inputs, list) or len(inputs) != 2:
#         #             raise ValueError("inputs must be a list of two strings of code.")
#         #         if not all(isinstance(code, str) for code in inputs):
#         #             raise ValueError("inputs must be a list of two strings of code.")
#         #         return {'code1': inputs[0]}, {'code2': inputs[1]}, {}

#         #     def preprocess(self, code1: str, code2: str, **kwargs):
#         #         return code1, code2

#         #     def _forward(self, code1, code2, **kwargs):
#         #         # Here you can implement the code similarity check logic
#         #         # using any suitable library or method.
#         #         # For example, you can use the difflib library:
#         #         import difflib
#         #         similarity_ratio = difflib.SequenceMatcher(None, code1, code2).ratio()
#         #         return {'similarity_ratio': similarity_ratio}

#         #     def postprocess(self, outputs, **kwargs):
#         #         return outputs['similarity_ratio']

#         # # from transformers import pipeline
#         # import torch
#         # from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
#         # from typing import List, Tuple
#         # from .base import PIPELINE_INIT_ARGS, Pipeline
#         # from ..utils import (
#         #     add_end_docstrings,
#         #     is_tf_available,
#         #     is_torch_available,
#         #     is_vision_available,
#         #     logging,
#         #     requires_backends,
#         # )

#         # logger = logging.get_logger(__name__)

#         # @add_end_docstrings(PIPELINE_INIT_ARGS)
#         # class CodeSimilarityPipeline(Pipeline):
#         #     """
#         #     Code similarity pipeline using any 'AutoModelForSequenceClassification'. This pipeline predicts the similarity between two code snippets.
#         #     ```python
#         #     >>> from transformers import pipeline

#         #     >>> classifier = pipeline(model="codebert-base")
#         #     >>> code1 = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
#         #     >>> code2 = "def fib(n):\n    if n <= 1:\n        return n\n    else:\n        return fib(n-1) + fib(n-2)"
#         #     >>> classifier(code1, code2)
#         #     'score': 0.563736617565155
#         #     ```
#         #     """

#         #     def __init__(self, **kwargs):
#         #         # super().__init__(*args, **kwargs)
#         #         # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         #         # self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
#         #         # self.code1 = ""
#         #         # self.code2 = ""
#         #     #     # self.save_pretrained('./code_similarity_pipeline')

#         #     # def __init__(self, *args, **kwargs):
#         #     #     super().__init__(*args, **kwargs)
#         #     #     self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer)
#         #     #     self.model = AutoModel.from_pretrained(self._model)
#         #         super().__init__(**kwargs)
#         #         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
#         #         self.model = AutoModel.from_pretrained("microsoft/codebert-base")

#         #     def _sanitize_parameters(self, inputs):
#         #         if isinstance(inputs, dict):
#         #             if "code1" not in inputs:
#         #                 raise ValueError("Parameter 'code1' is missing.")
#         #             if "code2" not in inputs:
#         #                 raise ValueError("Parameter 'code2' is missing.")
#         #             return inputs
#         #         elif isinstance(inputs, tuple) and len(inputs) == 2:
#         #             return {"code1": inputs[0], "code2": inputs[1]}
#         #         else:
#         #             raise ValueError("Invalid input format.")

#         #     # def _sanitize_parameters(self, inputs, **tokenizer_kwargs):
#         #     #     preprocess_params = tokenizer_kwargs

#         #     #     postprocess_params = {}
#         #     #     if isinstance(inputs, dict):
#         #     #         if "code1" not in inputs:
#         #     #             raise ValueError("Parameter 'code1' is missing.")
#         #     #         if "code2" not in inputs:
#         #     #             raise ValueError("Parameter 'code2' is missing.")
#         #     #         return inputs
#         #     #     elif isinstance(inputs, tuple) and len(inputs) == 2:
#         #     #         return {"code1": inputs[0], "code2": inputs[1]}
#         #     #     else:
#         #     #         raise ValueError("Invalid input format.")
#         #     #     return self._preprocess_params, self._forward_params, self._postprocess_params

#         #     def preprocess(self, inputs):
#         #         inputs = self._sanitize_parameters(inputs)
#         #         return inputs

#         #     def _forward(self, inputs):
#         #         input_str = f"Code 1: {inputs['code1']}\nCode 2: {inputs['code2']}\n"
#         #         input_encoding = self.tokenizer(
#         #             input_str,
#         #             return_tensors="pt",
#         #             truncation=True,
#         #             padding=True,
#         #             max_length=512,
#         #         )
#         #         input_encoding.to(self.model.device)
#         #         outputs = self.model(**input_encoding)
#         #         code1_embed, code2_embed = outputs.last_hidden_state.mean(dim=1)
#         #         similarity_score = self.similarity(code1_embed, code2_embed).item()
#         #         return similarity_score

#         #     def postprocess(self, outputs):
#         #         return outputs

#         #     def __call__(self, inputs):
#         #         inputs = self.preprocess(inputs)
#         #         outputs = self._forward(inputs)
#         #         outputs = self.postprocess(outputs)
#         #         return outputs

#         #     # def __init__(self, *args, **kwargs):
#         #     #     super().__init__(*args, **kwargs)
#         #     #     self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
#         #     #     self.model = AutoModel.from_pretrained("microsoft/codebert-base")
#         #     #     self.similarity = torch.nn.CosineSimilarity(dim=0)

#         #     # def _forward(self, inputs):
#         #     #     input_str = f"Code 1: {inputs['code1']}\nCode 2: {inputs['code2']}\n"
#         #     #     input_encoding = self.tokenizer(
#         #     #         input_str,
#         #     #         return_tensors="pt",
#         #     #         truncation=True,
#         #     #         padding=True,
#         #     #         max_length=512,
#         #     #     )
#         #     #     input_encoding.to(self.model.device)
#         #     #     outputs = self.model(**input_encoding)
#         #     #     code1_embed, code2_embed = outputs.last_hidden_state.mean(dim=1)
#         #     #     similarity_score = self.similarity(code1_embed, code2_embed).item()
#         #     #     return similarity_score

#         #     # def _sanitize_parameters(self, inputs):
#         #     #     if isinstance(inputs, dict):
#         #     #         if "code1" not in inputs:
#         #     #             raise ValueError("Parameter 'code1' is missing.")
#         #     #         if "code2" not in inputs:
#         #     #             raise ValueError("Parameter 'code2' is missing.")
#         #     #         return inputs
#         #     #     elif isinstance(inputs, tuple) and len(inputs) == 2:
#         #     #         return {"code1": inputs[0], "code2": inputs[1]}
#         #     #     else:
#         #     #         raise ValueError("Invalid input format.")

#         #     # def preprocess(self, inputs):
#         #     #     inputs = self._sanitize_parameters(inputs)
#         #     #     return inputs

#         #     # def postprocess(self, outputs):
#         #     #     return outputs

#         #     # def __call__(self, inputs):
#         #     #     inputs = self.preprocess(inputs)
#         #     #     outputs = self._forward(inputs)
#         #     #     outputs = self.postprocess(outputs)
#         #     #     return outputs

#         #     # def __init__(self, model='microsoft/codebert-base', **kwargs):
#         #     #     super().__init__(**kwargs)
#         #     #     self.model = model
#         #     #     self.tokenizer = AutoTokenizer.from_pretrained(self.model)
#         #     #     self.model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=2)
#         #     #     # self.save_pretrained('./code_similarity_pipeline')

#         #     # def _forward(self, inputs):
#         #     #     # Tokenize the input codes
#         #     #     encoded_text = self.tokenizer(inputs["code1"], inputs["code2"], padding=True, truncation=True, max_length=512, return_tensors='pt')

#         #     #     # Get the model's prediction for the input codes
#         #     #     with torch.no_grad():
#         #     #         logits = self.model(**encoded_text.to(self.device))[0]

#         #     #     return logits

#         #     # def postprocess(self, text1, text2):
#         #     #     inputs = {"text1": text1, "text2": text2}
#         #     #     logits = self._forward(inputs)
#         #     #     similarity_score = torch.softmax(logits, dim=1)[0][1].item()
#         #     #     print("Score:", similarity_score)
#         #     #     return similarity_score

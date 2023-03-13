import enum
import warnings

from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


# needed for t5 based models 
logger = logging.get_logger(__name__)

class ReturnType(enum.Enum):
    TENSORS = 0
    TEXT = 1

# 4 essential methods needed to implement preprocess, _forward, postprocess, and _sanitize_parameters.

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ProgrammingLanguageProcessing(Pipeline):
    """
    Programming Language Processing pipeline example. 
    More comment to add.
    """
    return_name = "PLP"

    def __init__(self, *args, **kwargs):
#        print("this is __init__ in programmingLanguageProcessing")
#        for key, value in kwargs.items():
#            print("%s == %s" % (key, value))
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
        self,
        return_tensors=None,
        max_length=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        **generate_kwargs
    ):
#        print("this is _sanitize_parameters in programmingLanguageProcessing")
#        for key, value in generate_kwargs.items():
#            print("%s == %s" % (key, value))
        forward_params = {}

        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if max_length is not None:
            forward_params["max_length"] = max_length

        preprocess_params = generate_kwargs 

        postprocess_params = generate_kwargs

        return preprocess_params, forward_params, postprocess_params

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    def _parse_and_tokenize(self, *args, truncation):
#        inputs = self.tokenizer(*args, padding=padding, truncation=truncation, return_tensors=self.framework)
        return inputs

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
#        print("this is preprocess in programmingLanguageProcessing")
#        for key, value in kwargs.items():
#            print("preprocess %s == %s" % (key, value))
        #inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        inputTensor = self.tokenizer(inputs, return_tensors="pt").input_ids
        return inputTensor

    def _forward(self, model_inputs, **generate_kwargs):
#        print("this is _forward in programmingLanguageProcessing")
#        print("model_inputs = %s" % model_inputs)
#        for key, value in generate_kwargs.items():
#            print("_forward %s == %s" % (key, value))
        max_length = 20
        if generate_kwargs["max_length"] is not None:
          max_length = generate_kwargs["max_length"]
        output_ids = self.model.generate(model_inputs, max_length=max_length)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
#        print("this is postprocess in programmingLanguageProcessing")
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        model_outputs["output_ids"][0],
                        skip_special_tokens=True,
                    )
                }
            records.append(record)
        return records


@add_end_docstrings(PIPELINE_INIT_ARGS)
class CodeSummarizationPipeline(ProgrammingLanguageProcessing):
    return_name = "Code summarization"




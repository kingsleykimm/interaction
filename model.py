from vllm import LLM


class MLModel:
    # need to change the weights
    def __init__(self, model_name="MolmoForCausalLM"):
        self.model = LLM(model_name)
    
    def infer(self, text):
        return self.model.generate(text)


class GestureRecognizer:
    pass

class 
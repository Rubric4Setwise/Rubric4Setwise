from rankify.dataset.dataset import Document

from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
from rankify.utils.pre_defined_methods import METHOD_MAP

class Reranking:
    """
    A class for reranking documents using different ranking models.

    Attributes:
        method (str): The name of the reranking method to be used.
        model (str): The name of the model used for reranking.
        api_key (str, optional): An optional API key for accessing remote models or services.
        ranker (BaseRanking): An instance of a ranking model used for reranking.
    """
    def __init__(self, method: str= None, model_name: str= None, api_key: str = None , **kwargs):
        """
        Initializes a Reranking instance.

        Args:
            method (str, optional): The name of the reranking method to be used. Defaults to None.
            model_name (str, optional): The name of the model to be used for reranking. Defaults to None.
            api_key (str, optional): An optional API key for accessing remote models or services. Defaults to None.
            **kwargs: Additional arguments passed to the reranking model.

        Example:
            ```python
            model = Reranking(method="listt5", model_name="listt5-base")
            ```
        """
        self.method = method
        self.model = model_name
        self.api_key= api_key
        self.kwargs = kwargs
        self.ranker = self.initialize()
    
    def initialize(self):
        """
        Initializes the ranking model based on the specified method and model name.

        Returns:
            BaseRanking: An instance of a ranking model.

        Raises:
            ValueError: If the specified ranking method is not supported.

        Example:
            ```python
            model = Reranking(method="listt5", model_name="listt5-base")
            ranker = model.initialize()
            ```
        """
        
        if self.method not in HF_PRE_DEFIND_MODELS:
            # Method not in predefined models dict, use model_name as-is
            model_name = self.model
        elif self.model not in HF_PRE_DEFIND_MODELS[self.method]:
            model_name= self.model
        else:
            model_name= HF_PRE_DEFIND_MODELS[self.method][self.model]

        if self.method in METHOD_MAP:
            return METHOD_MAP[self.method](method=self.method,model_name=model_name, api_key=self.api_key, **self.kwargs)
        else:
            raise ValueError(f"Ranking method {self.method} is not supported or installed. please install the full version of Rankify")

    def rank(self, documents: list[Document]):
        """
        Reranks a list of documents using the specified ranking model.

        Args:
            documents (list[Document]): A list of Document instances that need to be reranked.

        Returns:
            list[Document]: The reranked list of Document instances.

        Example:
            ```python
            question = Question("When did Thomas Edison invent the light bulb?")
            answers = Answer(["1879"])
            contexts = [
                Context(text="Lightning strike at Seoul National University", id=1),
                Context(text="Thomas Edison invented the light bulb in 1879", id=5)
            ]
            document = Document(question=question, answers=answers, contexts=contexts)
            model = Reranking(method="listt5", model_name="listt5-base")
            model.rank([document])
            ```
        """
        return self.ranker.rank(documents=documents)
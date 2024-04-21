from src.client.gpt import GPTClient
from typing import Dict, Any

class QueryHandler():
    """ 
    Class that handles inputted queries from the streamlit frontend
    """
    def __init__(self) -> None:
        self.llm_client=GPTClient()

        self.coalescence_prompt="""
        You are a chatbot built to respond regarding user inquires about datasets. You must provide a response with only the information
        provided. You will be given a dataset summary, a link to the dataset, and the user's initial query. Given all this information, 
        respond to the user's initial query appropriately. You should aim to be a friendly chatbot that is able so seamlessly respond to
        the user's question.
        """

    def coalesce_response(self, atlas_res: Dict[str, Dict[str, Any]], initial_query: str) -> str:
        """ 
        Given the provided summary, link to dataset, and initial query, this 
        function responds with a concise response for the user.
        
        atlas_res: a dict obeying the following schema: {dataset_id: {"summary": str, "links": str}}
        """

        response = self.llm_client.query(initial_query, sys_prompt=self.coalescence_prompt)

        return response

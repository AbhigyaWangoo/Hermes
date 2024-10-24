from src.client.gpt import GPTClient
from typing import Dict, Any, Tuple, List

class QueryHandler():
    """ 
    Class that handles inputted queries from the streamlit frontend
    """
    def __init__(self) -> None:
        self.llm_client=GPTClient()

        self.coalescence_prompt="""
        You are a chatbot built to respond regarding user inquires about datasets. You must provide a response with only the information
        provided. You will be given the name of the dataset, a dataset summary, and the user's initial query. Keep in mind that the users' query
        might be an example of an entry in a dataset, and might not be a natural language prompt. Given all this information, 
        explain why the provided data might help the user. You should aim to be a friendly chatbot that is able so seamlessly respond to
        the user's question.
        """

    def pretty_print(self, response: str, name: str, summary: str, link: str ) -> str:
        """
        return a Pretty print str out a response to the user
        """

        response_format="%s\n\n%s\n\nDataset summary: %s \nlink to dataset: %s\n\n"

        return response_format % (name, response, summary, link)

    def coalesce_response(self, atlas_res: List[Tuple[str, Dict[str, Any]]], initial_query: str) -> List[Dict[str, Any]]: # Ideally shouldve wrapped the response in a class. oops.
        """
        Provided with a response from mongodb atlas, response to the user appropriately
        """
        responses=[]
        for dataset in atlas_res:
            name=dataset[0]
            metadata=dataset[1]

            for dataval in metadata:
                print(dataval)

            link=metadata['links'] if 'links' in metadata else None
            summary=metadata['dataset_summary'].replace('\n',' ') if 'dataset_summary' in metadata else None

            response = self.coalesce_a_response(initial_query, summary)

            response_json = {
                "response": response,
                "name": name,
                "summary": summary,
                "link": link
            }
            responses.append(response_json)

        return responses

    def coalesce_a_response(self, initial_query: str, summary: str) -> str:
        """ 
        Given the provided summary, and initial query, this 
        function responds with a concise response for the user.
        """

        final_prompt = f"initial query: {initial_query}\nsummary of found dataset: {summary}"

        response = self.llm_client.query(final_prompt, sys_prompt=self.coalescence_prompt)

        return response

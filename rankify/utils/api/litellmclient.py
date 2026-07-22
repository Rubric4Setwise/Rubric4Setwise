
from litellm import completion

class LitellmClient:
    #  https://github.com/BerriAI/litellm
    def __init__(self, keys=None):
        self.api_key = keys

    def chat(self, return_text=True, *args, **kwargs):
        response = completion(api_key=self.api_key, *args, **kwargs)
        if return_text:
            response = response.choices[0].message.content
        return response
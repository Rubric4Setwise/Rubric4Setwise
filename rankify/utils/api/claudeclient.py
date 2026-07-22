from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


class ClaudeClient:
    def __init__(self, keys, url):
        
        self.anthropic = Anthropic(api_key=keys, base_url=url)

    def chat(self, messages, return_text=True, max_tokens=300, *args, **kwargs):
        system = ' '.join([turn['content'] for turn in messages if turn['role'] == 'system'])
        messages = [turn for turn in messages if turn['role'] != 'system']
        if len(system) == 0:
            system = None
        completion = self.anthropic.beta.messages.create(messages=messages, system=system, max_tokens=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.content[0].text
        return completion

    def text(self, max_tokens=None, return_text=True, *args, **kwargs):
        completion = self.anthropic.beta.messages.create(max_tokens_to_sample=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.completion
        return completion

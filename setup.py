from typing import List
import openai

openai.api_key = "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"


class Demo(object):
    def __init__(self, engine, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of

    def get_multiple_sample(self, prompt_list: List[str]):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of
        )
        results = [choice.text for choice in response.choices]
        return results


def run():
    demo = Demo(
        engine="text-davinci-002",  # text-davinci-002: best, text-ada-001: lowest price
        temperature=0.2,  # control randomness: lowring results in less random completion (0 ~ 1.0)
        max_tokens=256,  # max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # control diversity (0 ~ 1.0)
        frequency_penalty=0.2,  # how to penalize new tokens based on their existing frequency (0 ~ 2.0)
        presence_penalty=0.2,  # 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率 (0 ~ 2.0)
        best_of=3  # 这个是说从多少个里选最好的，如果这里是10，就会生成10个然后选最好的，但是这样会更贵(1 ~ 20)
    )
    results = demo.get_multiple_sample(
            ["how to solve this equation: (x^3)+1=9? Let's think step by step."]
    )
    print(results[0])


if __name__ == '__main__':
    run()

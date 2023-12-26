from transformers import pipeline, set_seed
import os
import json
from tqdm import tqdm


def generate_examples(prompt_list, model_name='gpt2', max_length=50, num_return_sequences=2, seed=42):
    import os
    import torch
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    generator = pipeline('text-generation', model=model_name, device="cpu")
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        example = {'prompt': prompt}
        for i, res in enumerate(result):
            answer = res['generated_text'].lstrip().removeprefix(prompt).strip()
            example[f'answer{i + 1}'] = answer
        examples.append(example)
        print(json.dumps(example, indent=2))
    return examples

if __name__ == "__main__":
    prompts = [
        "What is the latest news on the stock market?",
        "What is the current state of the economy?",
        "What are the latest developments in technology?",
        "What is the political situation in the Middle East?",
        "What are the latest trends in fashion and beauty?",
        "What are the top travel destinations for this year?",
        "What are some healthy recipes for a vegan diet?",
        "What are the most important events happening in the world today?",
        "What are some tips for improving mental health?",
    ]
    generate_examples(prompts)
    
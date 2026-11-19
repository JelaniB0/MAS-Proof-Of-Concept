import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://researchwork.openai.azure.com/openai/v1/"
)

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

def generate_answer(question, deployment=None):
    selected_deployment = deployment or DEPLOYMENT

    response = client.responses.create(
        model=selected_deployment,
        input=question,
        max_output_tokens=500
    )
    answer = response.output_text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return answer, input_tokens, output_tokens

if __name__ == "__main__":
    answer, input_tokens, output_tokens = generate_answer(
        "What is 15 multiplied by 12? Give just the answer."
    )
    print("Answer:", answer)
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)

if __name__ == "__main__":
    for deployment in ["gpt-5-nano", "gpt-4.1-mini", "gpt-5.1"]:
        answer, input_tokens, output_tokens = generate_answer(
            "What is 12 + 15? Give only the answer.",
            deployment=deployment
        )
        print(deployment, answer, input_tokens, output_tokens)
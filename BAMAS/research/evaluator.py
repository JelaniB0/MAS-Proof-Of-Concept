import re
import math
def extract_final_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None
    # the explicit GSM8K final-answer marker
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None



def evaluate_answer(model_answer, reference_answer):
    modelAns = extract_final_answer(model_answer)
    referenceAns = extract_final_answer(reference_answer)
    if modelAns is None or referenceAns is None:
        return 0
    cleanedModelAns = modelAns.replace(",", "")  
    cleanedReferenceAns = referenceAns.replace(",","")
    if cleanedModelAns == cleanedReferenceAns:
        return 1
    else:
        return 0
    

def process_sample(sample, model_answer):
    question = sample["question"]
    reference_answer = sample["answer"]
    correctness = evaluate_answer(model_answer,reference_answer)
    return correctness

def calculate_reward(correctness, cost, budget, cost_weight):
    costNormalized = cost / budget
    reward = correctness - cost_weight * costNormalized
    return reward

def calculate_cost(input_tokens, output_tokens, input_price, output_price):

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost
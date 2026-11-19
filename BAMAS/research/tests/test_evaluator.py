from evaluator import (
    extract_final_answer,
    evaluate_answer,
    calculate_cost,
    calculate_reward
)
test_cases = [
    ("16 - 3 - 4 = 9\n#### 18", "18"),
    ("The answer is 42.", "42"),
    ("I calculated 10, then 20. #### 30", "30"),
    ("", None),
    ("No numerical answer here.", None),
    ("The answer is -12.5. #### -12.5", "-12.5"),
    ("The total is 1,500 dollars. #### 1500", "1500"),
    ("I considered 20 options. The answer is 7.", "7"),
    ("The answer is 0.75.", "0.75"),
    ("There is no answer.", None),
]
for response, expected in test_cases:
    actual = extract_final_answer(response)
    print(actual, "Expected:", expected)
    assert actual == expected


#testing
print(extract_final_answer("10 + 5 = 15. Therefore, the answer is 12."))
print(extract_final_answer("The answer is 1,250."))
print(extract_final_answer("I could not solve this question."))
## limitation in extracting answer from string - assuming last number to be final answer 

comparison_tests = [
    ("The answer is 18.", "#### 18", 1),
    ("The answer is 1,500.", "#### 1500", 1),
    ("The answer is -12.5.", "#### -12.5", 1),
    ("The answer is 42.", "#### 24", 0),
    ("No numerical answer.", "#### 18", 0),
    ("The answer is 0.75.", "#### 0.75", 1),
]

for response, reference, expected in comparison_tests:
    actual = evaluate_answer(response, reference)
    print(actual, "Expected:", expected)
    assert actual == expected
#testing
print(evaluate_answer("10+2 = 12. Therefore, the answer is 12.", "#### 12"))
print(evaluate_answer("The answer is 1,000.", "#### 1000"))
print(evaluate_answer("The answer is 15.", "#### 12"))
print(evaluate_answer("I don't know.", "#### 12"))

print(calculate_reward(1, 0.002, 0.01, 0.1))
print(calculate_reward(1, 0.008, 0.01, 0.1))
print(calculate_reward(0, 0.002, 0.01, 0.1))
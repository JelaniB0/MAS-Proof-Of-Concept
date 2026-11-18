# Fixed-checkpoint difficulty evaluation

Repeated evaluation of an already inspected held-out split; not new independent validation.

No model or threshold training took place. Lower MSE/MAE is better; higher Spearman means better difficulty ranking.

## Recorded pool-outcome accuracy

Unknown targets are excluded separately for each pool.

| Dataset | Questions | Global-mean MSE | Family-mean MSE | Predictor MSE |
| --- | ---: | ---: | ---: | ---: |
| grade-school-math | 1095 | 0.092619 | 0.077436 | 0.073355 |
| mbpp | 61 | 0.187209 | 0.158490 | 0.150679 |

| Pool | Observed targets | MAE | RMSE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| weak | 744 | 0.2656 | 0.3206 | 0.301 |
| intermediate | 778 | 0.2373 | 0.2906 | 0.255 |
| strong | 888 | 0.1600 | 0.2232 | 0.248 |

Paired MSE reduction intervals (positive favors predictor):

- global_mean: 0.020750, 95% interval [0.016314655083414575, 0.02505449237814218].
- dataset_mean: 0.004325, 95% interval [0.0005403338467925062, 0.007958329790036151].

Per-dataset comparisons against the family-mean baseline:

- grade-school-math: MSE reduction 0.004081; 95% interval [0.0008983926473860199, 0.007089560073814231].
- mbpp: MSE reduction 0.007811; 95% interval [-0.022162807466146725, 0.03644627376658019].

## External questions: predictions without pool-outcome labels

MATH, HumanEval, official GSM8K test and authored prompts lack fixed-pool outcome labels. Predictions alone cannot establish difficulty accuracy.

All exact normalized-question matches to training, validation or the inspected test split are excluded.

| Dataset | Evaluated | Excluded | Within similarity threshold | Mean weak | Mean intermediate | Mean strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K-official-test | 1319 | 0 | 99.7% | 0.4507 | 0.3239 | 0.1114 |
| MATH | 5000 | 0 | 53.2% | 0.5735 | 0.4962 | 0.3404 |
| HumanEval | 164 | 0 | 78.7% | 0.6584 | 0.5075 | 0.3853 |
| new-questions | 20 | 0 | 45.0% | 0.4770 | 0.4214 | 0.2766 |

MATH difficulty-level association (proxy only):

{"weak": 0.27166856266578304, "intermediate": 0.23926080918274859, "strong": 0.23025014755361375}

| MATH human level | Questions | Mean weak | Mean intermediate | Mean strong |
| --- | ---: | ---: | ---: | ---: |
| 1 | 437 | 0.4821 | 0.4164 | 0.2699 |
| 2 | 894 | 0.5221 | 0.4527 | 0.2986 |
| 3 | 1131 | 0.5558 | 0.4779 | 0.3178 |
| 4 | 1214 | 0.5981 | 0.5197 | 0.3633 |
| 5 | 1324 | 0.6310 | 0.5461 | 0.3903 |

A dataset mean is a prediction distribution, not proof of accuracy. Similarity acceptance is not confidence.

## New-question probes

These are qualitative sanity checks, not labeled accuracy measurements.

The set includes the previously inspected `what's 1 + 1` debugging prompt; it is not a fresh confirmatory test.

| Prompt | Weak | Intermediate | Strong | Supported |
| --- | ---: | ---: | ---: | --- |
| what's 1 + 1 | 0.6708 | 0.7427 | 0.5947 | False |
| What is 1 + 1? | 0.5289 | 0.6331 | 0.3949 | False |
| Add one and one. | 0.6445 | 0.6091 | 0.2625 | False |
| Maya has one pear and receives another pear. How many pears does Maya have? | 0.3162 | 0.2040 | 0.0819 | True |
| Compute 1 + 1. Return only the number. | 0.6334 | 0.6060 | 0.4552 | False |
| Compute 1 + 1 and briefly explain your answer. | 0.8407 | 0.8227 | 0.6478 | False |
| What is 17 + 26? | 0.3246 | 0.3318 | 0.1537 | True |
| Calculate 89 minus 34. | 0.3625 | 0.2219 | 0.1202 | False |
| What is three quarters plus two thirds? | 0.3607 | 0.3481 | 0.1021 | True |
| Solve 7x - 5 = 44 for x. | 0.5279 | 0.4499 | 0.1347 | True |
| Find all real roots of x^4 - 5x^2 + 4 = 0. | 0.2463 | 0.2162 | 0.1327 | False |
| Prove that there are infinitely many primes congruent to 3 modulo 4. | 0.2354 | 0.3153 | 0.2520 | False |
| Write a Python function that returns the square of an integer. | 0.3773 | 0.1277 | 0.1129 | True |
| Write a Python function that checks whether parentheses, brackets, and braces are balanced. | 0.5455 | 0.3029 | 0.1574 | True |
| Implement edit distance in Python using dynamic programming and explain its complexity. | 0.8858 | 0.8110 | 0.7234 | True |
| Implement red-black tree insertion in Python, including rotations and rebalancing. | 0.8386 | 0.5558 | 0.4017 | False |
| What is the capital of France? | 0.0957 | 0.1043 | 0.0235 | True |
| Which integer is larger, 9 or 4? | 0.3961 | 0.3955 | 0.2466 | True |
| Explain why a photon can carry momentum despite having zero rest mass. | 0.5034 | 0.4076 | 0.2873 | False |
| Write a warm two-sentence thank-you note to a helpful neighbor. | 0.2060 | 0.2222 | 0.2471 | False |

## Sample labeled questions

Five examples per family, sampled with seed 42 before inspecting their errors.

### grade-school-math.dev.6175

If Layla scored 104 goals in four hockey games and Kristin scored 24 fewer goals in the same four games, calculate the average number of goals the two scored.

Recorded target: `{"weak": 0.75, "intermediate": 0.75, "strong": 1.0}`
Predicted: `{"weak": 0.36304089426994324, "intermediate": 0.2848508954048157, "strong": 0.06093791127204895}`

### grade-school-math.dev.3984

Judson, Camilo, and Kenny decided to contribute money to paint their house. Judson contributed $500, Kenny contributed 20% more money than Judson, and Camilo contributed $200 more than Kenny. How much was the cost of painting the house?

Recorded target: `{"weak": 0.25, "intermediate": 0.0, "strong": 0.0}`
Predicted: `{"weak": 0.36754870414733887, "intermediate": 0.2801952660083771, "strong": 0.08092094957828522}`

### grade-school-math.dev.5369

Matt has a peanut plantation that is 500 feet by 500 feet.  1 square foot of peanuts can make 50 grams of peanuts.  If it takes 20 grams of peanuts to make 5 grams of peanut butter and 1 kg of peanut butter sells for $10 how much does he make from his plantation?

Recorded target: `{"weak": 0.75, "intermediate": 0.5, "strong": 0.33333333333333337}`
Predicted: `{"weak": 0.3045058846473694, "intermediate": 0.19165876507759094, "strong": 0.057916298508644104}`

### grade-school-math.dev.1659

In a national park, the number of redwoods is 20% more than the number of pines. If there are 600 pines in the national park, calculate the total number of pines and redwoods that are there.

Recorded target: `{"weak": 0.25, "intermediate": 0.25, "strong": 0.0}`
Predicted: `{"weak": 0.3625107407569885, "intermediate": 0.25685662031173706, "strong": 0.12650887668132782}`

### grade-school-math.dev.3940

Every 4 weeks, Helen hand washes her silk pillowcases.  It takes 30 minutes to hand wash all of them.  In 1 year, how much time does Helen spend hand washing her pillowcases?

Recorded target: `{"weak": 0.5, "intermediate": 0.75, "strong": null}`
Predicted: `{"weak": 0.3913538157939911, "intermediate": 0.2292439192533493, "strong": 0.07252474874258041}`

### mbpp.dev.72

Write a function to get the angle of a complex number.

Recorded target: `{"weak": 1.0, "intermediate": 0.75, "strong": 1.0}`
Predicted: `{"weak": 0.41607093811035156, "intermediate": 0.34326091408729553, "strong": 0.26712095737457275}`

### mbpp.dev.127

Write a function that takes in a list and an element and inserts the element before each element in the list, and returns the resulting list.

Recorded target: `{"weak": 0.5, "intermediate": 0.25, "strong": 0.0}`
Predicted: `{"weak": 0.4430924355983734, "intermediate": 0.18227098882198334, "strong": 0.08579929918050766}`

### mbpp.dev.239

Write a function that gives loss amount on a sale if the given amount has loss else return 0.

Recorded target: `{"weak": 0.5, "intermediate": 0.5, "strong": 1.0}`
Predicted: `{"weak": 0.5817495584487915, "intermediate": 0.4051734209060669, "strong": 0.35678645968437195}`

### mbpp.dev.383

Write a function to check whether the given month number contains 30 days or not. Months are given as number from 1 to 12.

Recorded target: `{"weak": 0.75, "intermediate": 0.0, "strong": 0.0}`
Predicted: `{"weak": 0.2851763963699341, "intermediate": 0.2740432918071747, "strong": 0.1604297012090683}`

### mbpp.dev.372

Write a function to add the given tuple to the given list.

Recorded target: `{"weak": 1.0, "intermediate": 0.75, "strong": 1.0}`
Predicted: `{"weak": 0.7541710734367371, "intermediate": 0.5860629677772522, "strong": 0.5055654048919678}`


MATH source: https://huggingface.co/datasets/EleutherAI/hendrycks_math
HumanEval source: https://github.com/openai/human-eval

Exact revisions and checksums are in protocol.json. Source solutions/tests were never used as predictor inputs or executed.
Every evaluated prediction is saved in the JSONL files. No paid LLM calls were made.

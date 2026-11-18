# Frozen MPNet experiment results

Existing test split has been inspected previously; this is an exploratory comparison, not a fresh confirmatory test.

Selection uses equal-weight dataset validation MSE. Test results do not select a model.

| Configuration | Validation dataset macro MSE | Validation question-weighted MSE |
| --- | ---: | ---: |
| task_natural_mse | 0.172185 | 0.166579 |
| task_sqrt_balanced_mse | 0.171787 | 0.165472 |

Selected: **task_sqrt_balanced_mse** (ensemble of 3 seeds).

Pool MSEs receive equal weight; missing pool targets are excluded independently.

| Model | Test pooled MSE | Test dataset macro MSE | rag_news MSE | rag_wiki MSE |
| --- | ---: | ---: | ---: | ---: |
| global_mean | 0.204792 | 0.205877 | 0.221675 | 0.190080 |
| dataset_mean | 0.197985 | 0.198433 | 0.204946 | 0.191919 |
| selected_frozen_ensemble | 0.186379 | 0.187144 | 0.198281 | 0.176006 |

Normalized tier-utility scenarios use invented [1,2,3] cost units. They are diagnostics, not API cost savings or BAMAS results.

A positive paired MSE reduction favors the selected model; intervals containing zero do not establish a clear advantage.

- Versus global_mean: question-weighted MSE reduction 0.018413; paired 95% interval [0.006573833396836316, 0.03246770203654232]. Dataset macro reduction 0.018733; interval [0.006542719117839547, 0.03348662244352843].
- Versus dataset_mean: question-weighted MSE reduction 0.011607; paired 95% interval [-0.004990958199852197, 0.028466145995433803]. Dataset macro reduction 0.011289; interval [-0.005455448590930375, 0.02755723133120713].

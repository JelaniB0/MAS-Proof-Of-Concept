# Frozen MPNet experiment results

Existing test split has been inspected previously; this is an exploratory comparison, not a fresh confirmatory test.

Selection uses equal-weight dataset validation MSE. Test results do not select a model.

| Configuration | Validation dataset macro MSE | Validation question-weighted MSE |
| --- | ---: | ---: |
| task_natural_mse | 0.085969 | 0.085219 |
| task_sqrt_balanced_mse | 0.085920 | 0.086631 |
| task_model_sqrt_mse | 0.085641 | 0.085842 |

Selected: **task_model_sqrt_mse** (ensemble of 3 seeds).

Pool MSEs receive equal weight; missing pool targets are excluded independently.

| Model | Test pooled MSE | Test dataset macro MSE | additional-evals MSE | arc-challenge MSE | grade-school-math MSE | hellaswag MSE | mbpp MSE | mmlu MSE | mtbench MSE | winogrande MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global_mean | 0.110930 | 0.124122 | 0.205382 | 0.103745 | 0.092619 | 0.089245 | 0.187209 | 0.123367 | 0.122877 | 0.068532 |
| dataset_mean | 0.101255 | 0.106804 | 0.136061 | 0.084251 | 0.077436 | 0.085249 | 0.158490 | 0.120164 | 0.125818 | 0.066963 |
| selected_frozen_ensemble | 0.086935 | 0.091953 | 0.045035 | 0.091890 | 0.073355 | 0.081175 | 0.150679 | 0.100333 | 0.122806 | 0.070348 |

Normalized tier-utility scenarios use invented [1,2,3] cost units. They are diagnostics, not API cost savings or BAMAS results.

A positive paired MSE reduction favors the selected model; intervals containing zero do not establish a clear advantage.

- Versus global_mean: question-weighted MSE reduction 0.023996; paired 95% interval [0.02198744523748675, 0.02607493912858392]. Dataset macro reduction 0.032169; interval [0.025198414866991318, 0.03955686051305912].
- Versus dataset_mean: question-weighted MSE reduction 0.014320; paired 95% interval [0.012447265227026022, 0.016189100319974323]. Dataset macro reduction 0.014851; interval [0.00865725278511641, 0.021023695024632333].

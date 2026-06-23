# Sensitivity Analysis Summary

## Shared Experimental Configuration

| Item | Setting |
|---|---|

| Algorithm | SkyFlow / decomposition only |

| Main/default setting | eta = 0.05, SLA/budget multiplier = 1.0 |

| Workflow-quality settings | W1-Q1, W1-Q2, W1-Q3, W2-Q1, W2-Q2, W2-Q3 |

| Initial active scenarios | fraction = 0.3, strategy = qbr |

| Initializer selection | disabled |

| Evaluation protocol | held-out evaluation |

| Metrics shown | expected cost and SVR |


## Eta Sensitivity: Aggregate by eta

|    Eta | Available settings   |   Mean cost |   Mean SVR (%) |   Optimal count |   Missing settings |
|-------:|:---------------------|------------:|---------------:|----------------:|-------------------:|
| 0.025  | 4/6                  |      4.8854 |          2.163 |               2 |                  2 |
| 0.0375 | 6/6                  |      9.611  |          3.076 |               2 |                  0 |
| 0.05   | 6/6                  |      9.7732 |          3.224 |               3 |                  0 |
| 0.0625 | 6/6                  |      9.7659 |          2.8   |               4 |                  0 |
| 0.075  | 6/6                  |      9.7659 |          2.894 |               4 |                  0 |
| 0.0875 | 6/6                  |      9.7659 |          2.896 |               4 |                  0 |
| 0.1    | 6/6                  |      9.611  |          2.803 |               6 |                  0 |


## Eta Sensitivity: Maximum deviation from eta=0.05 by setting

| Setting   | Valid eta points   | Max abs cost change vs eta=0.05 (%)   |   Eta at max cost change | Max abs SVR change vs eta=0.05 (pp)   |   Eta at max SVR change |
|:----------|:-------------------|:------------------------------------|-------------------------:|:------------------------------------|------------------------:|
| W1-Q1     | 7/7                | <0.01                               |                   0.0625 | 3.21                                |                  0.0625 |
| W1-Q2     | 7/7                | <0.01                               |                   0.025  | <0.01                               |                  0.025  |
| W1-Q3     | 6/7                | 2.90                                |                   0.0375 | 0.24                                |                  0.0375 |
| W2-Q1     | 7/7                | <0.01                               |                   0.0625 | 2.17                                |                  0.0375 |
| W2-Q2     | 7/7                | <0.01                               |                   0.0375 | 0.04                                |                  0.075  |
| W2-Q3     | 6/7                | 2.14                                |                   0.0375 | 0.14                                |                  0.0375 |


## SLA/Budget Multiplier Sensitivity: Aggregate feasible results

|   Multiplier |   Mean cost |   Mean cost change vs x1.0 (%) |   Mean SVR (%) |   Mean SVR change vs x1.0 (pp) | Optimal count   |
|-------------:|------------:|-------------------------------:|---------------:|-------------------------------:|:----------------|
|          1   |      9.7732 |                           0    |          3.224 |                          0     | 3/6             |
|          1.1 |      9.6109 |                          -0.84 |          0.018 |                         -3.206 | 6/6             |
|          1.2 |      9.6109 |                          -0.84 |          0     |                         -3.224 | 6/6             |
|          1.3 |      9.6109 |                          -0.84 |          0     |                         -3.224 | 6/6             |
|          1.4 |      9.6109 |                          -0.84 |          0     |                         -3.224 | 6/6             |
|          1.5 |      9.6109 |                          -0.84 |          0     |                         -3.224 | 6/6             |


## SLA/Budget Multiplier Sensitivity: Infeasible tightened cases

|   Multiplier | Failed/attempted   |
|-------------:|:-------------------|
|          0.8 | 6/6                |
|          0.9 | 6/6                |


## SLA/Budget Multiplier Sensitivity: Setting-level summary

| Setting   |   Cost x1.0 |   SVR x1.0 (%) |   Cost change x1.1 (%) |   SVR x1.1 (%) |   Cost change x1.2 (%) |   SVR x1.2 (%) | Cost/SVR x1.3-x1.5   |
|:----------|------------:|---------------:|-----------------------:|---------------:|-----------------------:|---------------:|:---------------------|
| W1-Q1     |      1.6868 |          6.22  |                   0    |          0     |                   0    |              0 | same as x1.2         |
| W1-Q2     |      6.0642 |          0.846 |                  -0    |          0.01  |                  -0    |              0 | same as x1.2         |
| W1-Q3     |     17.8079 |          4.334 |                  -2.9  |          0.038 |                  -2.9  |              0 | same as x1.2         |
| W2-Q1     |      3.3967 |          2.92  |                  -0.01 |          0     |                  -0.01 |              0 | same as x1.2         |
| W2-Q2     |      8.3943 |          0.806 |                  -0    |          0.03  |                  -0    |              0 | same as x1.2         |
| W2-Q3     |     21.2893 |          4.218 |                  -2.15 |          0.03  |                  -2.15 |              0 | same as x1.2         |


# exp_v16 Progress Summary (2026-02-07)

## Overall Top 10 by Test Blended ROI

| report                                            | algo   | objective   |   pair_n |   trio_n |   test_roi |   test_hit |   test_cov |   test_tickets |
|:--------------------------------------------------|:-------|:------------|---------:|---------:|-----------:|-----------:|-----------:|---------------:|
| reports/exp_v16_xgb_binary_betoff_b4_cov3.json    | xgb    | binary      |        0 |        0 |   110.15   |  0.195489  |  0.0385061 |            133 |
| reports/exp_v16_xgb_binary_value_pair1_trio2.json | xgb    | binary      |        1 |        2 |    90.0777 |  0.230059  |  0.689635  |          15184 |
| reports/exp_v16_debug_onecase.json                | lgbm   | binary      |        2 |        3 |    89.8277 |  0.0179554 |  0.99971   |          55194 |
| reports/exp_v16_lgbm_binary_hybrid2.json          | lgbm   | binary      |        2 |        3 |    89.8277 |  0.0179554 |  0.99971   |          55194 |
| reports/exp_v16_xgb_binary_winonly_b16_cov8.json  | xgb    | binary      |        0 |        0 |    87.1098 |  0.150289  |  0.0500869 |            173 |
| reports/exp_v16_xgb_binary_value_pair1_trio1.json | xgb    | binary      |        1 |        1 |    85.7195 |  0.205865  |  0.997105  |          12314 |
| reports/exp_v16_xgb_binary_value_pair2_trio2.json | xgb    | binary      |        2 |        2 |    85.0731 |  0.324459  |  0.696005  |          22223 |
| reports/exp_v16_xgb_binary_value_b64.json         | xgb    | binary      |        2 |        3 |    83.2108 |  0.324459  |  0.696005  |          36623 |
| reports/exp_v16_xgb_binary_betoff_b4.json         | xgb    | binary      |        2 |        0 |    83.1455 |  0.460625  |  1         |          20839 |
| reports/exp_v16_xgb_ranker_value_b64.json         | xgb    | ranker      |        3 |        3 |    80.7255 |  0.523531  |  0.990446  |          62150 |

## Practical Filter (test_cov>=0.50, test_tickets>=10000, test_hit>=0.20)

| report                                            | algo   | objective   |   pair_n |   trio_n |   test_roi |   test_hit |   test_cov |   test_tickets |
|:--------------------------------------------------|:-------|:------------|---------:|---------:|-----------:|-----------:|-----------:|---------------:|
| reports/exp_v16_xgb_binary_value_pair1_trio2.json | xgb    | binary      |        1 |        2 |    90.0777 |   0.230059 |   0.689635 |          15184 |
| reports/exp_v16_xgb_binary_value_pair1_trio1.json | xgb    | binary      |        1 |        1 |    85.7195 |   0.205865 |   0.997105 |          12314 |
| reports/exp_v16_xgb_binary_value_pair2_trio2.json | xgb    | binary      |        2 |        2 |    85.0731 |   0.324459 |   0.696005 |          22223 |
| reports/exp_v16_xgb_binary_value_b64.json         | xgb    | binary      |        2 |        3 |    83.2108 |   0.324459 |   0.696005 |          36623 |
| reports/exp_v16_xgb_binary_betoff_b4.json         | xgb    | binary      |        2 |        0 |    83.1455 |   0.460625 |   1        |          20839 |
| reports/exp_v16_xgb_ranker_value_b64.json         | xgb    | ranker      |        3 |        3 |    80.7255 |   0.523531 |   0.990446 |          62150 |
| reports/exp_v16_xgb_ranker_value_b24.json         | xgb    | ranker      |        2 |        3 |    79.8032 |   0.477417 |   1        |          52185 |
| reports/exp_v16_xgb_binary_value_b24.json         | xgb    | binary      |        3 |        3 |    74.7634 |   0.538506 |   1        |          62239 |
| reports/exp_v16_xgb_ranker_nomarket_b24.json      | xgb    | ranker      |        2 |        3 |    69.7855 |   0.348096 |   0.653735 |          34830 |
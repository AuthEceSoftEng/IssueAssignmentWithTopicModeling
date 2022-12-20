# IssueAssignmentWithTopicModeling
This repository contains all code and instructions required to reproduce the findings of the paper:  
```
Themistoklis Diamantopoulos, Nikolaos Saoulidis, and Andreas Symeonidis
"Automated Issue Assignment using Topic Modeling on Jira Issue Tracking Data"
Paper submitted at the IET Software journal.
```

## Instructions for running the code

### Retrieving the data
First, you have to download the dataset found [here](https://doi.org/10.5281/zenodo.5665895).  
Then, you must set the properties found in file `properties.py`.  
After that, you can run script `1_mongo_get_and_save_data.py` to retrieve the data.  

### Running the methodology
You can run the methodology by executing the files `2_features_preprocess_and_transform.py`, `3_text_preprocessing.py`, `4_prepare_train_test_sets.py`, `5_apply_classification.py`, `6_optimize_topics.py`.  
The results can be found in the data folder that you set up in the `properties.py` file as zip files. Each step produces different output files with the same numbers (e.g. step `3_text_preprocessing.py` produces files of the form `3_{project_name}_{num_assignees}_assignees.csv`).

### Generating the tables and figures
You can run the scripts `7_1_evaluation_auc.py`, `7_2_evaluation_num_assignees.py`, `7_3_evaluation_assignees.py`, `7_4_evaluation_labels.py`, and `7_5_evaluation_classifiers.py` to reproduce the tables and graphs shown in the paper.  


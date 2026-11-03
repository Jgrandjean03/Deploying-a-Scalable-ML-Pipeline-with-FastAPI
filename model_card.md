# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed as student coursework (WGU/Udacity), November 2026. The starter code was provided by the Udacity course repository.

The model is a scikit-learn `GradientBoostingClassifier`, which builds an ensemble of shallow decision trees one after another so that each tree corrects the errors of the previous ones. It predicts a binary label, whether a person's income is `>50K` or `<=50K`.

The hyperparameters are set as named constants in `ml/model.py`. The model uses 100 trees (`N_ESTIMATORS = 100`), a learning rate of 0.1 (`LEARNING_RATE = 0.1`), a maximum tree depth of 3 (`MAX_DEPTH = 3`), and a random seed of 42 (`RANDOM_STATE = 42`). Every other parameter is left at the scikit-learn default, and no hyperparameter tuning was performed.

The preprocessing is done by the provided `process_data` function in `ml/data.py`. The 8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex, and native-country) are one-hot encoded with a `OneHotEncoder` that is fit on the training split only and ignores unseen categories. The label is binarized with a `LabelBinarizer`. The 6 continuous features (age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week) are passed through without scaling. The value "?" is kept as its own category in the data rather than being cleaned or removed.

The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl`. The environment used Python 3.10.22, scikit-learn 1.5.1, pandas 2.2.2, and numpy 2.2.6.

## Intended Use

The model is intended as a teaching example of a complete machine learning pipeline, which covers training, evaluation on data slices, and serving predictions through an API. It is meant to be read and run by course reviewers and learners who want to see how such a pipeline fits together.

The model is not intended to make or support decisions about real people. It should not be used for hiring, lending, housing, insurance, benefits eligibility, or any other setting where a wrong prediction could harm someone. It is also not intended to estimate anyone's current income, because the training data comes from a 1994 census extract.

## Training Data

The data is the file `data/census.csv`, which is the Census Income dataset provided by the course. The course cites it as: Kohavi, R. (1996). Census Income [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.

The UCI page states that the data was donated on 4/30/1996. The UCI dataset notes state that the extraction was done by Barry Becker from the 1994 Census database. The notes also state that only records meeting the following conditions were extracted: `((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))`. This card does not expand those abbreviations, because the UCI notes quoted here do not define them.

The repository's `census.csv` has 32,561 rows and 15 columns, made up of 14 input features and the label column `salary`. The row count matches the training-file size given in the UCI notes (train=32561). The repository does not document how its copy of the file was produced.

The label is imbalanced. Of the 32,561 rows, 24,720 (75.92%) are labeled `<=50K` and 7,841 (24.08%) are labeled `>50K`, so a model that always predicted `<=50K` would be correct about 76% of the time. The value "?" appears in three columns, with 1,836 rows in workclass, 1,843 rows in occupation, and 583 rows in native-country.

The model was trained on 80% of the rows, which is 26,048 rows. The split was stratified on the label with the random seed 42 (`SPLIT_SEED` in `train_model.py`), so the share of `>50K` is the same in both splits.

## Evaluation Data

The evaluation data is the remaining 20% of the rows, which is 6,513 rows. This test split was held out before training and was processed with the encoder and label binarizer fitted on the training split. In the test split, 24.07% of the rows are labeled `>50K`, which is close to the 24.08% share in the training split. It comes from the same source file as the training data, so it has the same provenance and the same limitations.

## Metrics

Three metrics are used, and each is defined as follows: Precision is the share of people the model predicts as `>50K` who truly earn more than 50K. Recall is the share of people who truly earn more than 50K that the model finds. F1 is the harmonic mean of precision and recall, so it is high only when both are high.

On the full test split of 6,513 rows, the model reached a precision of 0.7881, a recall of 0.6237, and an F1 of 0.6963. In plain terms, about 79% of the people the model flags as `>50K` really are, and about 38% of the people who truly earn more than 50K are missed (1 minus the recall of 0.6237). The model therefore misses more high earners than it falsely flags. This follows from its precision (0.7881) being higher than its recall (0.6237): it tends to predict `>50K` only when it is fairly confident, so some people who truly earn more than 50K are not flagged.

The same three metrics were also computed on slices of the test data. A slice is the subset of test rows that share one value of a categorical feature, such as `sex: Female`. The function `performance_on_categorical_slice` was run for every value of the 8 categorical features, which gives 101 slices. Small slices give unstable numbers, so this card only compares slices with a Count of at least 100 test rows.

Three features show especially large gaps between slices. For occupation, the lowest F1 is 0.2759 for Other-service (684 rows) and the highest is 0.8213 for Prof-specialty (818 rows). For education, the lowest F1 is 0.0000 for 7th-8th (120 rows) and the highest is 0.8696 for Masters (318 rows). For marital-status, the lowest F1 is 0.4593 for Divorced (883 rows) and the highest is 0.7217 for Married-civ-spouse (2,977 rows). These gaps show that the single overall F1 of 0.6963 hides very uneven performance across groups. The results for all 101 slices, including the features not discussed here, are in `slice_output.txt` in the repository root.

Sex and race behave differently. For sex, the F1 values are close, at 0.6893 for Female and 0.6975 for Male, a gap of 0.0082. The error pattern still differs, because the Female slice has a higher precision (0.8503 against 0.7784) and a lower recall (0.5796 against 0.6319) than the Male slice. For race, among the three slices with a Count of at least 100, the F1 is 0.7006 for White, 0.6711 for Black, and 0.6408 for Asian-Pac-Islander, a gap of 0.0598 between the highest and the lowest. The Black slice has a precision of 0.8644 and a recall of 0.5484, so the model misses a larger share of true high earners there than in the White slice, which has a recall of 0.6291. The Amer-Indian-Eskimo slice (73 rows) has an F1 of 0.4286, but it is below the 100-row threshold. The Other race slice has only 45 rows and shows 1.0000 for all three metrics, which is too small to interpret.

The function `compute_model_metrics` calls scikit-learn with `zero_division=1`, so a metric that is mathematically undefined is printed as 1.0000. This means a printed 1.0000 can be an artifact and not a sign of good performance. The effect is measured in the 7th-8th education slice, which has 120 test rows: 5 of those people truly earn more than 50K, the model predicted 0 positives, and none were found. The recall is therefore 0.0000, and the precision of 1.0000 is the `zero_division` fill because no positives were predicted. The same happens in the Married-AF-spouse slice of marital-status, where 3 of the 5 rows are truly `>50K` and the model predicted none.

Slices with a Count below 20 are unreliable. There are 35 such slices, made up of 2 in workclass, 1 in education, 1 in marital-status, 1 in occupation, and 30 in native-country. Two of them, Never-worked and Without-pay in workclass, contain a single test row each and show 1.0000 for all three metrics. Their numbers should not be read as evidence of how the model treats those groups.

## Ethical Considerations

The model uses sensitive attributes as inputs. Race, sex, native-country, marital-status, and relationship are all part of the feature set, so the model can learn and reproduce patterns tied to those attributes. The slice results show that the model finds high earners at different rates for different groups, for example a recall of 0.5796 for women against 0.6319 for men, and 0.5484 for Black people against 0.6291 for White people. This card reports those figures but does not claim that the model is fair or unfair, and no bias mitigation was attempted.

The `>50K` threshold refers to 1994-era income, so it is not a statement about what counts as a high income today. The data is also not a complete picture of the population. The extraction conditions quoted above mean that only records meeting those conditions are included, so not everyone is represented. The UCI notes describe the source as the 1994 Census database, and this repository does not document anything further about the people covered or how the sample was drawn.

The value "?" is kept as a category, so a missing workclass, occupation, or native-country is treated as information by the model. This can act as a proxy for other characteristics, and the data does not explain why those values are missing.

## Caveats and Recommendations

The overall scores come from one 80/20 split and one training run, with no cross-validation, so no confidence intervals are available and the numbers could shift with a different split. The hyperparameters are the scikit-learn defaults and were not tuned, so better performance may be possible.

The class imbalance of about 24% `>50K` may contribute to the model missing about 38% of the people who truly earn more than 50K, although this was not tested.

Slice results should be read with care. Slices with fewer than 20 test rows are unreliable, and the `zero_division=1` setting can print 1.0000 for an undefined metric, even in a slice with 120 rows. Gaps between slices are large for occupation, education, and marital-status, and the model should not be assumed to work equally well for every group.

The data is from a 1994 census extract with filters applied, and the repository does not document how its copy of the file was produced. For these reasons the model should not be used to make decisions about individuals. Before any real use, the data should be replaced or validated against current data, the slice metrics should be reviewed for the groups affected, and a fairness review should be completed.

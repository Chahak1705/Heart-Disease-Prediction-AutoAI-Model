import warnings
warnings.filterwarnings('ignore')

# Install required packages (run these in terminal if not already installed)
# pip install ibm-watsonx-ai autoai-libs~=2.0 scikit-learn==1.3.* lale~=0.8.3 snapml==1.14.*

from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation
import numpy as np
from sklearn.pipeline import make_pipeline, make_union
from autoai_libs.transformers.exportable import (
    NumpyColumnSelector, CompressStrings, NumpyReplaceMissingValues,
    NumpyReplaceUnknownValues, boolean2float, CatImputer, CatEncoder,
    float32_transform, FloatStr2Float, NumImputer, OptStandardScaler,
    NumpyPermuteArray
)
from autoai_libs.cognito.transforms.transform_utils import TAM, FS1
import autoai_libs.cognito.transforms.transform_extras
from sklearn.cluster import FeatureAgglomeration
from snapml import SnapSVMClassifier

# ── Data Connections ──────────────────────────────────────────────────────────

training_data_references = [
    DataConnection(
        data_asset_id='7d585c30-f65f-4154-87a4-910c07c8c4a1'
    ),
]

training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/2f1e12ce-c0fa-4735-9b61-a85d7df87403/wml_data/1389bcf4-a54a-4092-aa54-dbdbc6622e5b/data/automl',
        model_location='auto_ml/2f1e12ce-c0fa-4735-9b61-a85d7df87403/wml_data/1389bcf4-a54a-4092-aa54-dbdbc6622e5b/data/automl/model.zip',
        training_status='auto_ml/2f1e12ce-c0fa-4735-9b61-a85d7df87403/wml_data/1389bcf4-a54a-4092-aa54-dbdbc6622e5b/training-status.json'
    )
)

# ── Experiment Metadata ───────────────────────────────────────────────────────

experiment_metadata = dict(
    prediction_type='binary',
    prediction_column='target',
    holdout_size=0.1,
    scoring='accuracy',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    deployment_url='https://au-syd.ml.cloud.ibm.com',
    project_id='f6c3aea8-8f13-4b76-99f2-9eef909d8de3',
    positive_label=1,
    drop_duplicates=True,
    include_batched_ensemble_estimators=[],
    feature_selector_mode='auto'
)

# ── Authentication ────────────────────────────────────────────────────────────

import getpass
api_key = getpass.getpass("Please enter your api key (press enter): ")

from ibm_watsonx_ai import Credentials, APIClient

credentials = Credentials(api_key=api_key, url=experiment_metadata['deployment_url'])
client = APIClient(credentials)

if 'space_id' in experiment_metadata:
    client.set.default_space(experiment_metadata['space_id'])
else:
    client.set.default_project(experiment_metadata['project_id'])

training_data_references[0].set_client(client)

X_train, X_test, y_train, y_test = training_data_references[0].read(
    experiment_metadata=experiment_metadata,
    with_holdout_split=True,
    use_flight=True
)

# ── Pipeline P8: Snap SVM Classifier ─────────────────────────────────────────

# Categorical columns branch
numpy_column_selector_0 = NumpyColumnSelector(columns=[1, 2, 5, 6, 8, 10, 11, 12])
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=["float_int_num"] * 8,
    missing_values_reference_list=["", "-", "?", float("nan")],
    misslist_list=[[], [], [], [], [], [], [], []],
)
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(filling_values=100001, missing_values=[])
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=100001.0,
    filling_values_list=[100001] * 8,
    missing_values_reference_list=["", "-", "?", float("nan")],
)
cat_imputer = CatImputer(missing_values=100001, sklearn_version_family="1", strategy="most_frequent")
cat_encoder = CatEncoder(
    dtype=np.float64, handle_unknown="error",
    sklearn_version_family="1", encoding="ordinal", categories="auto"
)
pipeline_0 = make_pipeline(
    numpy_column_selector_0, compress_strings, numpy_replace_missing_values_0,
    numpy_replace_unknown_values, boolean2float(), cat_imputer, cat_encoder, float32_transform()
)

# Numeric columns branch
numpy_column_selector_1 = NumpyColumnSelector(columns=[0, 3, 4, 7, 9])
float_str2_float = FloatStr2Float(
    dtypes_list=["float_int_num", "float_int_num", "float_int_num", "float_int_num", "float_num"],
    missing_values_reference_list=[],
)
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(filling_values=float("nan"), missing_values=[])
num_imputer = NumImputer(missing_values=float("nan"), strategy="median")
opt_standard_scaler = OptStandardScaler(use_scaler_flag=False)
pipeline_1 = make_pipeline(
    numpy_column_selector_1, float_str2_float, numpy_replace_missing_values_1,
    num_imputer, opt_standard_scaler, float32_transform()
)

# Merge branches
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(
    axis=0, permutation_indices=[1, 2, 5, 6, 8, 10, 11, 12, 0, 3, 4, 7, 9]
)

col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
             "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
col_dtypes = [np.dtype("float32")] * 13

# Feature transforms
tam_0 = TAM(
    tans_class=autoai_libs.cognito.transforms.transform_extras.IsolationForestAnomaly,
    name="isoforestanomaly", col_names=col_names, col_dtypes=col_dtypes
)
fs1_0 = FS1(cols_ids_must_keep=range(0, 13), additional_col_count_to_keep=12, ptype="classification")
tam_1 = TAM(
    tans_class=FeatureAgglomeration(),
    name="featureagglomeration",
    col_names=col_names + ["isoforestanomaly_0"],
    col_dtypes=col_dtypes + [np.dtype("int64")]
)
fs1_1 = FS1(cols_ids_must_keep=range(0, 13), additional_col_count_to_keep=12, ptype="classification")

# Final estimator
snap_svm_classifier = SnapSVMClassifier(
    class_weight="balanced",
    device_ids=np.array([0], dtype=np.uint32),
    fit_intercept=False,
    gamma=98.60256297110465,
    kernel="linear",
    max_iter=110,
    n_components=10,
    random_state=33,
    regularizer=6.332000612561614,
)

# Full pipeline
pipeline = make_pipeline(union, numpy_permute_array, tam_0, fs1_0, tam_1, fs1_1, snap_svm_classifier)

# ── Train & Evaluate ──────────────────────────────────────────────────────────

from sklearn.metrics import get_scorer

scorer = get_scorer(experiment_metadata['scoring'])
pipeline.fit(X_train.values, y_train.values.ravel())
score = scorer(pipeline, X_test.values, y_test.values)
print(f"Accuracy: {score}")
print("Sample predictions:", pipeline.predict(X_test.values[:5]))

# ── Store Model ───────────────────────────────────────────────────────────────

model_metadata = {
    client.repository.ModelMetaNames.NAME: 'P8 - Pretrained AutoAI pipeline'
}
stored_model_details = client.repository.store_model(
    model=pipeline,
    meta_props=model_metadata,
    experiment_metadata=experiment_metadata
)
print(stored_model_details)

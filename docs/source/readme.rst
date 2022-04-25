papAI package
##########################


**papAI package** is a Python package used to provide a user-friendly interface to use papAI's tools in a python interpreter.
This document describes how to use our package and its main functionnalities.


Requirements
------------

The **papAI package** requires Python 3.8 or newer.

You need an environment with installed:

pandas>=1.2.5
pydantic>=1.8.1
category_encoders>=2.2.2
imblearn
lightgbm==3.2.1
shap==0.39.0
rfpimp==1.3.7
pdpbox==0.2.0
xgboost==1.3.3

**tutorial on how to install this environment inc**

Installation
------------

Install from Test PyPi

.. code-block:: text

    pip install -i https://test.pypi.org/simple/ papai

Data
----

In order to create your pipeline and train it, you've to provide a training dataset.

Accepted format are ".parquet" and ".csv"

Give your training settings
---------------------------

In papai package you can create an experiment among these 3 tasks : Regression, Binary Classification and Multiclass Classification


.. parsed-literal::

  {
    "dataset_path": string,
    "preprocessing_config": see schema of preprocessing_config_,
    "model": {
        "features": list(string),
        "params": see schema of regression_models_,
        "targets": {
            "target_transformer": string,
            "target_name": string
        }
    }
    "data_segregation": {
        "kind": string,
        "nb_groups": int
    }
  }


Regression task
^^^^^^^^^^^^^^^

Here we specify all models available for a regression task:

.. _regression_models:

.. code-block:: python

    class AdaBoost:
        base_estimator: str = None
        n_estimators: int = 50
        learning_rate: float = 1.0
        loss: LossEnum = LossEnum.linear
        random_state: int = None

    class DecisionTree:
        criterion: CriterionEnum = CriterionEnum.gini
        splitter: SplitterEnum = SplitterEnum.best
        max_depth: int = None
        min_samples_split: Union[int, float] = 2
        min_samples_leaf: Union[int, float] = 1
        min_weight_fraction_leaf: float = 0.0
        max_features: Union[int, float, str] = None
        random_state: int = None
        max_leaf_nodes: int = None
        min_impurity_decrease: float = 0.0
        class_weight: Union[dict, List[dict], str] = None
        ccp_alpha: PositiveFloat = 0.0

    class ElasticNetModel:
        alpha: float = 1.0
        l1_ratio: float = 0.5
        fit_intercept: bool = True
        normalize: bool = False
        precompute: bool = False
        max_iter: int = 1000
        copy_X: bool = True
        tol: float = 1e-4
        warm_start: bool = False
        positive: bool = False
        random_state: int = None
        selection: SelectionEnum = SelectionEnum.cyclic

    class KNeighbors:
        n_neighbors: int = 5
        weights: Union[WeightsEnum, Callable] = WeightsEnum.uniform
        algorithm: AlgorithmEnum = AlgorithmEnum.auto
        leaf_size: int = 30
        p: int = 2
        metric: Union[str, Callable] = "minkowski"
        metric_params: Optional[dict] = None
        n_jobs: int = None

    class LarsModel:
        fit_intercept: bool = True
        verbose: bool = False
        normalize: bool = True
        precompute: Union[bool, str] = "auto"
        n_nonzero_coefs: int = 500
        eps: float = np.finfo(float).eps
        copy_X: bool = True
        fit_path: bool = True
        jitter: float = None
        random_state: int = None

    class LassoLarsModel:
        alpha: float = 1.0
        fit_intercept: bool = True
        verbose: bool = False
        normalize: bool = True
        precompute: Union[bool, str] = "auto"
        max_iter: int = 500
        eps: float = np.finfo(float).eps
        copy_X: bool = True
        fit_path: bool = True
        positive: bool = False
        jitter: float = None
        random_state: int = None

    class LassoModel:
        alpha: float = 1.0
        fit_intercept: bool = True
        normalize: bool = False
        precompute: bool = False
        max_iter: int = 1000
        copy_X: bool = True
        tol: float = 1e-4
        warm_start: bool = False
        positive: bool = False
        random_state: int = None
        selection: SelectionEnum = SelectionEnum.cyclic

    class Linear:
        fit_intercept: bool = True
        normalize: bool = False
        copy_X: bool = True
        n_jobs: int = None
        positive: bool = False

    class MLP:
        hidden_layer_sizes: List[int] = (100,)
        activation: ActivationEnum = ActivationEnum.relu
        solver: SolverEnum = SolverEnum.adam
        alpha: float = 1e-4
        batch_size: Optional[int] = "auto"
        learning_rate: LearningRateEnum = LearningRateEnum.constant
        learning_rate_init: float = 0.001
        power_t: float = 0.5
        max_iter: int = 200
        shuffle: bool = True
        random_state: int = None
        tol: float = 1e-4
        verbose: bool = False
        warm_start: bool = False
        momentum: float = 0.9
        nesterovs_momentum: bool = True
        early_stopping: bool = False
        validation_fraction: float = 0.1
        beta_1: float = 0.9
        beta_2: float = 0.999
        epsilon: float = 1e-8
        n_iter_no_change: int = 10
        max_fun: int = 15000

    class OrthogonalMatchingPursuitModel(BaseModel):
        n_nonzero_coefs: int = None
        tol: float = None
        fit_intercept: bool = True
        normalize: bool = True
        precompute: Union[bool, str] = "auto"

    class RandomForest:
        n_estimators: int = 100
        criterion: CriterionEnum = CriterionEnum.mse
        max_depth: int = None
        min_samples_split: Union[int, float] = 2
        min_samples_leaf: Union[int, float] = 1
        min_weight_fraction_leaf: float = 0.0
        max_features: Union[MaxFeaturesEnum, int, float] = MaxFeaturesEnum.auto
        random_state: int = None
        max_leaf_nodes: int = None
        min_impurity_decrease: float = 0.0
        min_impurity_split: float = None
        bootstrap: bool = True
        oob_score: bool = False
        n_jobs: int = None
        verbose: int = 0
        warm_start: bool = False
        ccp_alpha: PositiveFloat = 0.0
        max_samples: Union[int, float] = None

    class RidgeModel:
        alpha: float = 1.0
        fit_intercept: bool = True
        normalize: bool = False
        max_iter: int = None
        copy_X: bool = True
        tol: float = 1e-3
        solver: SolverEnum = SolverEnum.auto
        random_state: int = None

    class SGD:
        loss: LossEnum = LossEnum.squared_loss
        penalty: PenaltyEnum = PenaltyEnum.l2
        alpha: float = 0.0001
        l1_ratio: float = 0.15
        fit_intercept: bool = True
        max_iter: int = 1000
        tol: float = 1e-3
        shuffle: bool = True
        verbose: int = 0
        epsilon: float = 0.1
        random_state: int = None
        learning_rate: str = "invscaling"
        eta0: float = 0.01
        power_t: float = 0.25
        early_stopping: bool = False
        validation_fraction: float = 0.1
        n_iter_no_change: int = 5
        warm_start: bool = False
        average: Union[bool, int] = False

    class XGBoost:
        n_estimators: int
        max_depth: Optional[int]
        learning_rate: Optional[float]
        verbosity: Optional[int]
        objective: Union[str, Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]]]
        booster: BoosterEnum
        tree_method: Optional[str] = "auto"
        n_jobs: Optional[int]
        gamma: Optional[float]
        min_child_weight: Optional[float]
        max_delta_step: Optional[float]
        subsample: Optional[float]
        colsample_bytree: Optional[float]
        colsample_bylevel: Optional[float]
        colsample_bynode: Optional[float]
        reg_alpha: Optional[float]
        reg_lambda: Optional[float]
        scale_pos_weight: Optional[float]
        base_score: Optional[float]
        random_state: Optional[int]
        missing: float = nan
        num_parallel_tree: Optional[int]
        monotone_constraints: Optional[Union[Dict[str, int], str]]
        interaction_constraints: Optional[Union[str, List[Tuple[str]]]]
        importance_type: ImportanceTypeEnum = ImportanceTypeEnum.gain


Classification task
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we specify all models available for a classification task:



.. code-block:: python

    class LogisticRegressionModel:
        penalty: PenaltyEnum = PenaltyEnum.l2
        dual: bool = False
        tol: float = 1e-4
        C: float = 1.0
        fit_intercept: bool = True
        intercept_scaling: float = 1
        class_weight: Union[dict, str] = None
        random_state: int = None
        solver: SolverEnum = SolverEnum.lbfgs
        max_iter: int = 100
        multi_class: MultiClassEnum = MultiClassEnum.auto
        verbose: int = 0
        warm_start: bool = False
        n_jobs: int = None
        l1_ratio: float = None

    class AdaBoost:
        base_estimator: str = None
        n_estimators: int = 50
        learning_rate: float = 1.0
        algorithm: AlgorithmEnum = AlgorithmEnum.SAMME_R
        random_state: int = None

Create your first papai pipeline
--------------------------------

.. code-block:: python

    from papai.pipeline import Pipeline

    # for json_train see training params spec section
    pipeline = Pipeline(**json_train)

    pipeline.get_dataset() # Check your dataset

    pipeline.initialize_pipeline() # Initialize your dataset


Preprocessing steps available
-----------------------------

.. _preprocessing_config:

::

  {
     "Transformers": [Scaler, Encoder],
     "Data_augmentation": [SMOTE, ROS]
  }

Train your model
----------------

From your previous created pipeline you can fit it:

.. code-block:: python

    pipeline.split_train_test() # First split your dataset into training and test sets.

    pipeline.add_estimator_step()

    pipeline.final_train()


Eval your model
----------------

From your previous created pipeline you can fit it:

.. code-block:: python

    # evaluate your model
    pipeline.eval_model()

Save your model
---------------

.. code-block:: python

    # saving your model (.joblib format) in the "PATH" provided
    pipeline.save(filepath="PATH")

Interpret your pipeline
-----------------------

.. code-block:: python

    pipeline.interpret_pipeline()

Use your pipeline to make prediction
------------------------------------

.. code-block:: python

    from papai.prediction import Prediction

    papai_predict = Prediction(**json_predict)

    papai_predict.predict()

External model
--------------

**inc**
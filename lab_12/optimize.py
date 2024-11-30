#%% IMPORTS
import os
import optuna
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle
import datetime
import pkg_resources
import json

#%%

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#%% MANAGE FOLDERS

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

#%% GET BEST MODEL

def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    runs_sorted = runs.sort_values("metrics.valid_f1", ascending=False)
    best_model_id = runs_sorted["run_id"].iloc[0]
    print("best_model_id: ", best_model_id)
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model

#%% MANAGE EXPERIMENTS
# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

#%% SETTING UP LOGGING WITH MLFLOW
# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html

# override Optuna's default logging to ERROR only
# optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


# def champion_callback(study, frozen_trial):
#     """
#     Logging callback that will report when a new trial iteration improves upon existing
#     best trial values.

#     Note: This callback is not intended for use in distributed computing systems such as Spark
#     or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
#     workers or agents.
#     The race conditions with file system state management for distributed trials will render
#     inconsistent values with this callback.
#     """

#     winner = study.user_attrs.get("winner", None)

#     if study.best_value and winner != study.best_value:
#         study.set_user_attr("winner", study.best_value)
#         if winner:
#             improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
#             print(
#                 f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
#                 f"{improvement_percent: .4f}% improvement"
#             )
#         else:
#             print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


#%% LOG LIBRARY VERSIONS

def log_library_versions():
    installed_packages = pkg_resources.working_set
    package_versions = {pkg.key: pkg.version for pkg in installed_packages}
    with open("library_versions.json", "w") as f:
        json.dump(package_versions, f, indent=4)
    mlflow.log_artifact("library_versions.json", "configs")

#%% GRADIENT BOOSTING WITH OPTUNA

def xgboost_optuna(data: pd.DataFrame, experiment_id):
    data = data.dropna()
    
    y = data['Potability']
    X = data.drop(['Potability'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, stratify=y
    )
    
    col_transformer = ColumnTransformer(
        transformers=[
            ('scaler', 
             StandardScaler(), 
             ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
              'Organic_carbon', 'Trihalomethanes', 'Turbidity']
            )
        ],
        remainder='passthrough'
    )
    
    # X_train_preprocessed = col_transformer.fit_transform(X_train)
    # X_test_preprocessed = col_transformer.transform(X_test)
    
    X = col_transformer.fit_transform(X)
    
    def objective(trial):
        gb_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        pipeline = Pipeline(steps=[
                ('preprocessor', col_transformer),
                ('classifier', GradientBoostingClassifier(**gb_params, random_state=42))
            ])
        run_name = (
            f"GB_Optuna_lr_{gb_params['learning_rate']}_"
            f"n_estimators_{gb_params['n_estimators']}_"
            f"max_depth_{gb_params['max_depth']}_"
            f"min_samples_split_{gb_params['min_samples_split']}_"
            f"min_samples_leaf_{gb_params['min_samples_leaf']}_"
            f"subsample_{gb_params['subsample']}_"
            f"max_features_{gb_params['max_features']}"
        )
    
        with mlflow.start_run(run_name=run_name, nested=True):
            # y_pred = classifier.predict(X_test_preprocessed)
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
            score = scores.mean()
            trial.report(score,step=1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # classifier.fit(X_train,y_train)
            pipeline.fit(X_train, y_train)
            mlflow.log_params(gb_params)
            mlflow.log_metric("valid_f1", score)
            mlflow.sklearn.log_model(
                pipeline, 
                artifact_path='model',
                input_example=X_train.head(1)
            )
    
        return score
    
    parent_run_name = f"GradientBoosting_Optuna_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # parent run
    with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name, nested=True):
        study = optuna.create_study(direction='maximize',
                                    sampler=TPESampler(),
                                    pruner=MedianPruner())
        study.optimize(
            objective, 
            timeout=60*60,
            show_progress_bar=True, 
            # callbacks=[champion_callback]
        )
        mlflow.log_params(study.best_params)
    
        # best_params = study.best_params
        
        fig_optimization_history = plot_optimization_history(study)
        fig_param_importances = plot_param_importances(study)
        fig_parallel = plot_parallel_coordinate(
            study, 
            params=[
                'learning_rate', 
                'n_estimators', 
                'max_depth', 
                'min_samples_split', 
                'min_samples_leaf', 
                'subsample', 
                'max_features'
            ]
        )
        
        fig_optimization_history.write_html("./plots/optimization_history.html")
        fig_param_importances.write_html("./plots/param_importances.html")
        fig_parallel.write_html("./plots/parallel_coordinate.html")
        
        mlflow.log_artifact("./plots/optimization_history.html", "plots")
        mlflow.log_artifact("./plots/param_importances.html", "plots")
        mlflow.log_artifact("./plots/parallel_coordinate.html", "plots")
    
    return study, experiment_id


#%% OPTIMIZE MODEL

def optimize_model():
    # mlflow.autolog()
    
    df = pd.read_csv(r'./water_potability.csv')
    
    experiment_name = "GradientBoosting_Potability"
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    
    study, experiment_id = xgboost_optuna(df, experiment_id)
    
    best_model = get_best_model(experiment_id)
    
    log_library_versions()
    
    with open('./models/best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    return best_model

#%% MAIN EXECUTION

if __name__ == "__main__":
    optimize_model()
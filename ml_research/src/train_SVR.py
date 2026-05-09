import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate # Using cross_validate for multiple metrics
from src import PipleLineStepBase
import logging

# Configure logger
logger = logging.getLogger(__name__)

class SuperconductivityTrainingStep(PipleLineStepBase):
    def __init__(self, X_train, X_test, y_train, y_test, experiment_name="Superconductivity_CV_Study"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.experiment_name = experiment_name
        
        logger.info(f"Initializing Training Step for experiment: {self.experiment_name}")
        
        try:
            mlflow.set_tracking_uri("http://localhost:5002")
            mlflow.set_experiment(self.experiment_name)
            logger.info("Connected to MLflow tracking server.")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            raise

    def objective(self, trial):
        # 1. Suggest Hyperparameters
        n_components = trial.suggest_int("pca_components", 19, 21)
        epsilon = trial.suggest_float("svr_epsilon", 0.01, 0.1, log=True)
        c_param = trial.suggest_float("svr_c", 3000.0, 10000.0, log=True)
        
        logger.info(f"Trial {trial.number} | PCA: {n_components}, C: {c_param:.2f}, Epsilon: {epsilon:.2f}")

        # 2. Define Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('svr', SVR(C=c_param, epsilon=epsilon))
        ])

        # 3. Cross-Validation (Multiple Metrics)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            try:
                # Using cross_validate to get both RMSE and R2 simultaneously
                scoring = {
                    'rmse': 'neg_root_mean_squared_error',
                    'r2': 'r2'
                }
                
                results = cross_validate(
                    pipeline, self.X_train, self.y_train, 
                    scoring=scoring, 
                    cv=cv, 
                    n_jobs=2
                )
                
                avg_rmse = -np.mean(results['test_rmse'])
                avg_r2 = np.mean(results['test_r2'])
                
                logger.info(f"Trial {trial.number} Results | CV RMSE: {avg_rmse:.4f}, CV R2: {avg_r2:.4f}")
                
                # Log metrics to MLflow
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_rmse", avg_rmse)
                mlflow.log_metric("cv_r2", avg_r2)
                
                return avg_rmse # Optuna minimizes RMSE
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise

    def run(self):
        logger.info("Starting Hyperparameter Optimization...")
        
        with mlflow.start_run(run_name="CV_SVR_PCA_Optimization"):
            # A. Optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=10)
            
            logger.info(f"Optimization finished. Best CV RMSE: {study.best_value:.4f}")

            # B. Final Training
            best_params = study.best_params
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=best_params['pca_components'])),
                ('svr', SVR(C=best_params['svr_c'], epsilon=best_params['svr_epsilon']))
            ])
            
            final_pipeline.fit(self.X_train, self.y_train)

            # C. Evaluation on Test Set
            test_preds = final_pipeline.predict(self.X_test)
            test_rmse = root_mean_squared_error(self.y_test, test_preds)
            test_r2 = r2_score(self.y_test, test_preds)
            
            # D. Registry and Logging
            logger.info(f"Final Test Metrics | RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
            
            mlflow.log_params(best_params)
            mlflow.log_metric("final_test_rmse", test_rmse)
            mlflow.log_metric("final_test_r2", test_r2)
            
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model",
                registered_model_name="superconductivity_svr_model"
            )
            
            return final_pipeline
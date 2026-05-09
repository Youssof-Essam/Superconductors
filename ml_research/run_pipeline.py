import logging
import sys
from src import LoadSplitData, SuperconductivityTrainingStep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def execute_data_ingestion(path, filename, target):
    try:
        logger.info(f"Phase 1: Ingesting and splitting {filename}")
        data_loader = LoadSplitData(
            path=path,
            filename=filename,
            target_column=target,
            test_size=0.2,
            random_state=42
        )
        return data_loader.run()
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

def execute_model_training(X_train, X_test, y_train, y_test):
    try:
        logger.info("Phase 2: Initiating optimization and training")
        trainer = SuperconductivityTrainingStep(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            experiment_name="Superconductivity_Production_Run"
        )
        return trainer.run()
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def run_pipeline():
    logger.info("Pipeline execution started")
    
    try:
        X_train, X_test, y_train, y_test = execute_data_ingestion(
            path="ml_research/data",
            filename="super_conductors_raw.csv",
            target="critical_temp"
        )
        
        final_model = execute_model_training(
            X_train, 
            X_test, 
            y_train, 
            y_test
        )
        
        logger.info("Pipeline execution completed successfully")
        return final_model

    except Exception as e:
        logger.critical(f"Pipeline terminated prematurely: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
import os

# Define the base directory where the structure will be created
base_directory = "e:/Projects/github plan/machine-learning-1-to-100"

# Define the folder and file structure
structure = {
    "Data_collection": {
        "Web_scraping": ["scraper.py", "scraping_tools.md", "data_cleaning.py"],
        "APIs": ["api_data_fetch.py", "api_docs.md"],
        "Datasets": [
            "download_datasets.py",
            "datasets_description.md",
            "sample_data/sample1.csv",
            "sample_data/sample2.json",
        ],
    },
    "Data_Preprocessing": {
        "Data_cleaning": [
            "clean_data.py",
            "remove_outliers.py",
            "preprocess_examples.md",
        ],
        "Data_transformation": [
            "feature_scaling.py",
            "encoding_categorical.py",
            "transformation_methods.md",
        ],
        "Data_splitting": ["train_test_split.py", "cross_validation.py"],
    },
    "Feature_engineering": {
        "Feature_creation": ["create_features.py", "feature_examples.md"],
        "Feature_selection": [
            "select_features.py",
            "correlation_matrix.py",
            "feature_importance.py",
        ],
        "Dimensionality_reduction": ["pca.py", "lda.py"],
    },
    "Model_preparation_practices": {
        "Model_training": [
            "train_models.py",
            "hyperparameter_tuning.py",
            "training_examples.md",
        ],
        "Model_evaluation": [
            "evaluate_models.py",
            "confusion_matrix.py",
            "evaluation_metrics.md",
        ],
        "Model_save_load": ["model_save.py", "model_load.py"],
        "Model_tracking": [
            "mlflow_tracking.py",
            "mlflow_ui_screenshot.md",
            "mlflow_experiments_config.md",
        ],
        "Model_version_control": [
            "dvc_pipeline_setup.md",
            "dvc_commands.py",
            "dvc_tracking_example.py",
        ],
    },
    "Neural_network": {
        "Feedforward_Network": ["nn_model.py", "nn_architecture.md"],
        "Convolutional_Network": ["cnn_model.py", "cnn_examples.md"],
        "Recurrent_Network": ["rnn_model.py", "rnn_examples.md"],
    },
    "Reinforcement_learning": {
        "Q_learning": ["qlearning.py", "qlearning_example.md"],
        "Deep_Q_Network": ["dqn_model.py", "dqn_examples.md"],
        "Policy_Gradient": ["policy_gradient.py", "policy_gradient_examples.md"],
    },
    "Supervised_learning": {
        "Regression": [
            "linear_regression.py",
            "ridge_regression.py",
            "regression_examples.md",
        ],
        "Classification": [
            "logistic_regression.py",
            "svm_model.py",
            "knn_model.py",
            "classification_examples.md",
        ],
        "Ensemble_methods": ["random_forest.py", "xgboost.py", "ensemble_examples.md"],
    },
    "Unsupervised_learning": {
        "Clustering": [
            "kmeans.py",
            "hierarchical_clustering.py",
            "clustering_examples.md",
        ],
        "Dimensionality_reduction": ["pca.py", "tsne.py", "dr_examples.md"],
        "Anomaly_detection": ["isolation_forest.py", "anomaly_examples.md"],
    },
    # Base files
    ".": ["README.md", "LICENSE", ".gitignore", "create_structure.py"],
}


def create_structure(base_path, structure):
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder) if folder != "." else base_path
        os.makedirs(folder_path, exist_ok=True)

        if isinstance(contents, list):
            for file in contents:
                file_path = os.path.join(folder_path, file)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if not os.path.exists(file_path):  # avoid overwriting existing files
                    with open(file_path, "w", encoding="utf-8") as f:
                        if file.endswith(".md"):
                            f.write(
                                f"# {file.replace('_', ' ').split('.')[0].title()}\n"
                            )
                        elif file.endswith(".py"):
                            f.write(f"# Script: {file}\n\n")
                        else:
                            f.write("")  # leave other files empty
        elif isinstance(contents, dict):
            create_structure(folder_path, contents)


# Start the creation process
create_structure(base_directory, structure)

print("✅ Directory structure created successfully!")

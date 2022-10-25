from kfp import dsl
import mlrun

@dsl.pipeline(name="kedro-pipeline",)
def pipeline(
    companies_path: str,
    shuttles_path: str,
    reviews_path: str,
    parameters: dict
):

    # Init MLRun project and functions
    project = mlrun.get_current_project()
    dp_fn = project.get_function("dp").apply(mlrun.mount_v3io())
    ds_fn = project.get_function("ds").apply(mlrun.mount_v3io())
    serving_fn = project.get_function("serving").apply(mlrun.mount_v3io())
    
    # Data processing
    preprocess_companies_run = project.run_function(
        function=dp_fn,
        handler="preprocess_companies",
        inputs={"companies" : companies_path},
        outputs=["preprocessed_companies"]
    )
    
    preprocess_shuttles_run = project.run_function(
        function=dp_fn,
        handler="preprocess_shuttles",
        inputs={"shuttles" : shuttles_path},
        outputs=["preprocessed_shuttles"]
    )
    
    create_model_input_table_run = project.run_function(
        function=dp_fn,
        handler="create_model_input_table",
        inputs={
            "companies" : preprocess_companies_run.outputs["preprocessed_companies"],
            "shuttles" : preprocess_shuttles_run.outputs["preprocessed_shuttles"],
            "reviews" : reviews_path
        },
        outputs=["model_input_table"]
    )
    
    # Model training and evaluation
    split_data_run = project.run_function(
        function=ds_fn,
        handler="split_data",
        inputs={"data" : create_model_input_table_run.outputs["model_input_table"]},
        params={"parameters" : parameters},
        outputs=["X_train", "X_test", "y_train", "y_test"]
    )
    
    train_run = project.run_function(
        function=ds_fn,
        handler="train_model",
        inputs={
            "X_train" : split_data_run.outputs["X_train"],
            "y_train" : split_data_run.outputs["y_train"]
        },
        outputs=["model"]
    )
    
    evaluate_run = project.run_function(
        function=ds_fn,
        handler="evaluate_model",
        inputs={
            "regressor" : train_run.outputs["model"],
            "X_test" : split_data_run.outputs["X_test"],
            "y_test" : split_data_run.outputs["y_test"]
        }
    )
    
    # Model deployment + monitoring
    serving_fn.set_tracking()
    deploy_run = project.deploy_function(
        function=serving_fn,
        models=[
            {
                "key": "regressor",
                "model_path": train_run.outputs["model"],
            }
        ],
    )
from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='cnneis-experiment')
    env = Environment.get(workspace=ws, name="AzureML-tensorflow-2.6-ubuntu20.04-py38-cuda11-gpu")
    dataset = Dataset.get_by_name(ws, name='dataset')
    config = ScriptRunConfig(
        source_directory='./src',
        script='cnn.py',
        compute_target='cnneiscc',
        environment=env,
        arguments=['--data_folder', dataset.as_mount()]
    )

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)
import os
import dotenv
import tarfile
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

def create_tar_gz(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

create_tar_gz("GNN_Deployment", "model.tar.gz")

sagemaker_session = sagemaker.Session()
dotenv.load_dotenv()
role = os.getenv("SAGEMAKER_EXECUTION_ROLE")

model_uri = sagemaker_session.upload_data("model.tar.gz", key_prefix="aml-gnn-model")

pytorch_model = PyTorchModel(
    model_data=model_uri,
    role=role,
    entry_point="inference.py",
    source_dir="GNN_Deployment/codes",
    framework_version="2.1.0",
    py_version="py310"
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072, 
    max_concurrency=10
)

predictor = pytorch_model.deploy(
    serverless_inference_config=serverless_config
)

print(f"Endpoint name: {predictor.endpoint_name}")
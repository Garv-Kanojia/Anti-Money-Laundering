import os
import torch
import pickle
import pandas as pd
from torch_geometric.data import HeteroData
import GNN_create

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(model_dir, 'model_artifacts', 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)
        
    dummy_graph = HeteroData()
    model = GNN_create.HMPNN_ct_3Layer(dummy_graph, node_type="account").to(device)
        
    model_path = os.path.join(model_dir, 'model_artifacts', 'aml_hmpnn_ct_3layer.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return {"model": model, "artifacts": artifacts, "device": device}

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return pd.read_json(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    device = model_dict["device"]
    
    # In a real scenario, this is where we format the incoming JSON 
    # into the PyG HeteroData subgraph format using our saved artifacts
    subgraph = HeteroData().to(device)
    
    with torch.no_grad():
        prediction = model(subgraph.x_dict, subgraph.edge_index_dict, subgraph.edge_attr_dict)
        
    return prediction.cpu().numpy().tolist()

def output_fn(prediction, accept):
    if accept == 'application/json':
        return {"fraud_probability": prediction}
    raise ValueError(f"Unsupported accept type: {accept}")
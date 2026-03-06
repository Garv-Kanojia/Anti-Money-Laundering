# Anti Money Laundering System

## Overview

Developed a money laundering fraud detector using a Heterogeneous Graph Neural Network (GNN) and Knowledge Graphs, combined through Agentic AI and deployed with AWS Bedrock AgentCore.

This system improves "smurfing" detection from **65% to 80%** using a Heterogeneous GNN over XGBoost, trained on the IBM AML dataset. The complete solution achieves an **AUC-ROC of 85%** by combining the GNN and AI agents.

## Features

- **Heterogeneous Graph Neural Network (GNN):** Uses a custom Heterogeneous Message Passing Neural Network (`HMPNN`) to classify accounts and detect fraudulent transaction patterns.
- **Multi-Agent Architecture:** Investigates fraud suspects using OSINT data and Knowledge Graphs. The system uses LangGraph to orchestrate a planner agent and multiple specialized analyzer agents (Tavily Web Search, Knowledge Graph, OpenSanctions, Corporate Registry, PostgreSQL).
- **Graph Storage:** Transactions are stored as graphs for GNN training and further analysis by AI agents in Neo4j.
- **Serverless Deployment:** The GNN model is packaged and deployed on AWS SageMaker Serverless Inference, while the multi-agent system is deployed using AWS Bedrock AgentCore.

## Repository Structure

- `Data_preprocessing.ipynb`: Cleans and preprocesses HI-Small account and transaction data. Generates daily metrics, applies target encoding and standard scaling, and constructs edge index and node features for PyTorch Geometric.
- `aml_gnn_train.ipynb`: Trains the `HMPNN_ct_3Layer` model using PyTorch Geometric on preprocessed graph data. Includes class imbalance handling using weighted BCE loss and evaluation for Youden-optimal threshold via ROC-AUC.
- `GNN_Deployment/`: Contains the inference script (`inference.py`), model definitions (`GNN_create.py`), and deployment requirements for SageMaker.
- `deployment_code.py`: Automates packaging GNN model artifacts (`model.tar.gz`) and deploying them to an AWS SageMaker Serverless endpoint.
- `Agent_deployment/`: Contains the LangGraph ReAct-style implementation (`Multi_Agent_code.py`) and the AWS Bedrock AgentCore entry point (`app.py`) for the investigative multi-agent system.

## Technologies Used

- **Machine Learning and Graphs:** PyTorch, PyTorch Geometric, Scikit-learn, Category Encoders, Pandas, Neo4j
- **Agentic AI:** LangGraph, custom LLM integrations
- **Cloud and Deployment:** AWS Bedrock AgentCore, AWS SageMaker Serverless Inference

## Acknowledgements and References

The GNN model architecture in `GNN_Deployment/codes/GNN_create.py` is based on the Heterogeneous MPNN implementation by [fredjo89](https://github.com/fredjo89).

- Project reference: https://github.com/fredjo89/heterogeneous-mpnn
- Paper reference: https://arxiv.org/abs/2307.13499

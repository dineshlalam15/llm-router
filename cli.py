import argparse
from llm_router.data.generator import generate_routing_data
# from llm_router.training.trainer import train_models (Trainer logic merges embedding and model.fit)

def main():
    parser = argparse.ArgumentParser(description="Marketing LLM Router CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("generate-data", help="Download HF datasets and generate routing labels")
    subparsers.add_parser("train", help="Train SVM and KNN models from processed data")
    
    route_parser = subparsers.add_parser("route", help="Test routing for a query")
    route_parser.add_argument("--query", required=True, type=str)
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_routing_data()
    elif args.command == "train":
        print("Training models... (Executes SentenceTransformer encoding -> SVM.fit() -> KNN.fit() -> joblib.dump())")
        # train_models()
    elif args.command == "route":
        from llm_router.routing.engine import DynamicRouter
        router = DynamicRouter()
        print(router.route(args.query))

if __name__ == "__main__":
    main()
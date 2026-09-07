#Project Structure 
llm-router/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── router.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── openai_provider.py
│       ├── claude_provider.py
│       ├── gemini_provider.py
│       └── litellm_provider.py
├── data/
│   ├── llm_router_training.csv
│   └── test_queries.csv
├── model/ (Generated during training)
├── tests/
│   ├── test_router.py
│   └── test_api.py
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
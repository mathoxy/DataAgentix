from smolagents import CodeAgent, LiteLLMModel

# Tools
from tools.data_tools import dataset_summary_tool, clean_dataset_tool, visualize_data_tool
from tools.doc_tools import generate_pdf_report
from tools.ml_tools import train_models, predict_with_model

# Models
data_model = LiteLLMModel(model_id="ollama/qwen3-coder:480b-cloud", provider="ollama")
ml_model = LiteLLMModel(model_id="ollama/qwen3-coder:480b-cloud", provider="ollama")
doc_model = LiteLLMModel(model_id="ollama/qwen3-coder:480b-cloud", provider="ollama")
orchestrator_model = LiteLLMModel(model_id="ollama/qwen3-coder:480b-cloud", provider="ollama")

# Data Agent
data_agent = CodeAgent(
    name="data_agent",
    model=data_model,
    tools=[
        dataset_summary_tool,
        clean_dataset_tool,
        visualize_data_tool,
    ],
    additional_authorized_imports = [
        "datasets",               # load_dataset
        "pandas",                 # pd
        "sklearn.impute",         # SimpleImputer
        "sklearn.model_selection",# train_test_split
        "matplotlib",             # matplotlib principal
        "matplotlib.pyplot",      # pyplot submodule
        "seaborn",                # sns
        "os"                      # os for mkdir etc.
    ]
    ,
    description="""
        A data analysis agent that can load datasets from files or the Hugging Face Hub, 
        provide summaries, clean data, and visualize datasets using various plots.
        It is designed to assist with data preprocessing and exploratory data analysis tasks.

        ## Available tools:
        - dataset_summary_tool(dataset_path: str) -> dict
            Example: dataset_summary_tool(dataset_path='data/raw/mydata.csv')
        - clean_dataset_tool(dataset_path: str) -> str
            Example: clean_dataset_tool(dataset_path='data/raw/mydata.csv')
            the cleaned dataset will be saved and the path returned as data/cleaned/mydata.csv
        - visualize_data_tool(dataset_path: str, target: str=None) -> list
            Example: visualize_data_tool(dataset_path='data/raw/mydata.csv', target='y')

        You MUST use these tools for all data loading, cleaning, summary, and visualization tasks. Do not write manual code for these operations. You MUST NOT create, modify, or save any files except through the provided tools.
    """
)    

# ML Agent
ml_agent = CodeAgent(
    name="ml_agent",
    model=ml_model,
    tools=[
        train_models,
        predict_with_model
    ],
    additional_authorized_imports = [
        "pandas",                 # pd
        "sklearn.model_selection",# train_test_split
        "sklearn.linear_model",   # LinearRegression, LogisticRegression
        "sklearn.ensemble",       # RandomForestClassifier, RandomForestRegressor
        "sklearn.metrics",        # accuracy_score, classification_report
        "joblib",                 # joblib for model saving/loading
        "os"                      # os for path operations
    ]
    ,
    description="""
        A machine learning agent that can build and evaluate machine learning models 
        using an automated pipeline. It can handle both classification and regression tasks,
        providing performance metrics for the trained models.

        ## Available tools:
        - train_models(cleaned_dataset_path: str, target: str, task_type: str) -> dict
            Example: train_models(cleaned_dataset_path='data/cleaned/mydata.csv', target='y', task_type='regression')
        - predict_with_model(model_path: str, input_data: dict) -> any
            Example: predict_with_model(model_path='models/linear_regression_model.joblib', input_data={'x': 20})

        You MUST use these tools for all training and prediction tasks. Do not write manual training or prediction code. You MUST NOT create, modify, or save any files except through the provided tools.
    """
)

doc_agent = CodeAgent(
    name="doc_agent",
    model=doc_model,
    tools=[generate_pdf_report],
    description="""
        A documentation agent that generates PDF reports summarizing data analysis and machine learning results.
        It compiles text sections and images into a structured PDF document.
        ## Available tools:
        - generate_pdf_report(title: str, sections: list, images: list) -> str
            Example: generate_pdf_report(
                title='Analysis Report',
                sections=[{"title": "Introduction", "content": "This is the introduction."}],
                images=['plots/figure1.png']
            )
        YOU MUST include the report path in your final response.
        
    """
)

orchestrator_agent = CodeAgent(
    name="orchestrator_agent",
    model=orchestrator_model,
    tools=[],
    managed_agents=[
        data_agent,
        ml_agent,
        doc_agent
    ],
    description="""
        An orchestrator agent that coordinates between the data analysis and preprocessing agent and the machine learning agent.
        It manages the workflow of cleaning, visualizing data, and subsequently training and evaluating ML models, as well as generating PDF reports.

        ## Managed agents:
        - data_agent: for cleaning, summary, and visualization (see its tools above)
        - ml_agent: for training and prediction (see its tools above)
        - doc_agent: for generating PDF reports (see its tools above)

        You MUST delegate tasks to the appropriate agent and use their tools for all operations. Do not write manual code for these steps.

        Once data is cleaned through the data_agent, you MUST use the path returned by the clean_dataset_tool for all subsequent operations
        The cleaned dataset is in data/cleaned/ folder. EXAMPLE: data/cleaned/mydata.csv

        The images used in the PDF report must be exactly those returned by the visualize_data_tool of the data_agent. Do not use any other images or sources.

        At the end of the workflow, provide a concise response summarizing the results.
        When there is a PDF report generated, return the path to the PDF file in your response as follows:
        {"report_path": "reports/file_name.pdf"}
        You MUST NOT use any other key, synonym, or format; always use 'report_path' exactly as shown above.
        Any other key or format will be ignored by the user interface.
        The value must be a valid, existing file path to the generated PDF.
    """
)

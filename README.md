# DataAgentix

**DataAgentix** is a simple collaborative AI-agent platform designed to automate real-world **data science workflows** using intelligent agents built with **Hugging Face SmolAgents**.

The platform orchestrates multiple specialized agents to transform raw data into insights, machine learning models, and simple reports end to end.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Gradio Interface](#gradio-interface)
- [Key Concepts](#key-concepts)
- [Future Improvements](#future-improvements)

## Project Objectives

- Apply and master **Small Agents (SmolAgents)** from Hugging Face
- Design **collaborative AI agents** with clear responsibilities
- Combine **built-in Hugging Face tools** and **custom tools**
- Improve skills in:
  - AI agent orchestration
  - Data science workflows
  - Tool-based agent design
  - Modular Python project architecture

## Architecture Overview

DataAgentix is composed of four main agents:

### Orchestrator Agent
Coordinates the full workflow and delegates tasks to specialized agents based on user intent.

### Data Agent
Responsible for data-related tasks:
- Dataset summary and inspection
- Data cleaning and preprocessing
- Automatic visualizations

### ML Agent
Handles machine learning tasks:
- Model training 
- Model evaluation (metrics, validation)

### Documentation Agent
Generates structured reports:
- Summarizes analysis and results
- Integrates generated plots
- Exports reports as **PDF files**



## Tech Stack
- Python 3.10+
- uv (package & environment manager)
- Hugging Face SmolAgents
- pandas, scikit-learn
- matplotlib, seaborn
- Gradio (interactive UI)
- ReportLab (report generation)

## Installation & Setup
1️⃣ Clone the repository
```bash
git clone https://github.com/mathoxy/DataAgentix.git
cd DataAgentix
```
2️⃣ Initialize environment with uv
```bash
uv sync
```
3️⃣ Run the application
```bash
uv run main.py
```
### Example User Prompt
```text
I have a CSV dataset about car prices.
Please analyze the data, clean it, visualize key patterns,
train a regression model to predict prices,
evaluate the model performance,
and generate a PDF report with all results.
The Orchestrator Agent will automatically:

Invoke the Data Agent for analysis and visualization

Call the ML Agent for modeling and evaluation

Ask the Doc Agent to generate a final report
```
## Gradio Interface
DataAgentix includes a Gradio UI that allows users to:

- Upload their own datasets
- Interact with agents via chat
- Download generated reports

## Key Concepts
- Agent orchestration
- Tool-based reasoning
- Secure code execution (authorized imports)
- Modular agent design
- Human-in-the-loop AI workflows

## Future Improvements
- Multi-dataset support
- Agent memory and state persistence
- Advanced AutoML tools
- Model deployment agent
- Dataset versioning
- Cloud execution support

***⭐ If you like this project
Give it a ⭐ on GitHub and feel free to contribute!***
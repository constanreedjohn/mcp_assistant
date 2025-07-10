# **MCP Application**

This is a locally hosted Chatbot with MCP integrated.

## Overall features:
* Locally hosted LLM: The chatbot uses Ollama LLM model and deploy on local machine
* MCP Client and Server: MCP tools are built in MCP server and FastMCP integrated with LLM as MCP Client.
* Image Gen Tool: Integrate locally deployed Image Generation model as an API endpoint into MCP tools for functionality.

## Setups
1. **LLM with llama-server**

Make sure you have llama-server installed.
* Pull the model

`llama-server -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M --port 4001 --jinja`

2. **Install Dependencies**

This project uses Conda environment.

```
conda create -n <your_env_name> python=3.11
pip install -r requirements.txt
```

3. Startup Image Gen tool:

In order to have the Image Generation tool, we will start the FastAPI server with the model endpoint.

Open a terminal
```
cd server/
python run main_api.py
```

The server will start on `http://localhost:3001`

4. Startup MCP Server:

We will startup the MCP Server with all of the tools.

Currently, there are 4 tools:
* **get_alerts:** Call a request to an external API for weather alert based on the US state.

* **get_forecast**: Call a request to an external API for weather alert based on the latitude and longtitude.

* **get_multiply**: Do multiplicate between 2 numbers.

* **generate_image**: Call to the hosted FastAPI Image Generation model.

Open a second terminal
```
cd server/
python main_mcp.py
```
The server will start on `http://localhost:5001`

5. Startup the main app:

This project uses Gradio as the UI of the application which has a chatbot interface along with uploading and displaying images from the tools.

Open a third terminal
```
cd server/
python app.py
```
The server will start on `http://localhost:7860`
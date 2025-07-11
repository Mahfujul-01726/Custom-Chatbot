# QnA Chatbot With Gradio

This is a QnA Chatbot built with Gradio.

## Hugging Face Space

You can find a live demo of this chatbot on Hugging Face Spaces here: [Chatbot - a Hugging Face Space by mahfuj735](https://huggingface.co/spaces/mahfuj735/Chatbot)


## Description
This is a simple Question and Answer chatbot built using Python and the Gradio library. It allows users to interact with a chatbot interface to ask questions and receive answers.

## Features
* Interactive web interface using Gradio
* Simple question and answer functionality
* Integration with OpenAI's GPT models
* Streaming responses
* Conversation memory
* Customizable system prompts

## Installation
To set up the project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/QnA-Chatbot-With-Gradio.git
    cd QnA-Chatbot-With-Gradio
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key**:
    Create a `.env` file in the root directory of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Usage
To run the chatbot, execute the `app.py` file:

```bash
python app.py
```

Once the application is running, open your web browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860/`).

## Code Quality and Maintainability Enhancements

Here are some suggestions to further enhance the code quality and maintainability of the `app.py` file:

1.  **Error Handling**: Implement more specific error handling for API calls (e.g., `openai.APIError`, `openai.RateLimitError`). This can provide more informative feedback to the user and improve the robustness of the application.

2.  **Configuration Management**: Instead of hardcoding model names and system prompts, consider loading them from a configuration file (e.g., `config.json` or `config.yaml`). This makes it easier to manage and update these values without modifying the code.

3.  **Logging**: Integrate a proper logging mechanism (e.g., Python's `logging` module) instead of just printing errors to the console. This helps in debugging and monitoring the application in production.

4.  **Code Structure and Modularity**: For larger applications, consider breaking down `app.py` into smaller, more manageable modules. For example, API interactions could be in `api_client.py`, UI components in `ui_components.py`, and utility functions in `utils.py`.

5.  **Asynchronous Operations**: While `AsyncOpenAI` is used, ensure all I/O bound operations are truly asynchronous to prevent blocking the event loop. For example, file operations like `json.dump` in `export_conversation` are synchronous and could be made asynchronous if performance becomes a bottleneck in a highly concurrent environment.

6.  **Type Hinting**: Continue to use and expand type hints throughout the codebase. This improves code readability, helps catch errors during development, and makes it easier for others to understand the expected types of arguments and return values.

7.  **Docstrings**: Ensure all functions and classes have clear and concise docstrings explaining their purpose, arguments, and return values. This is crucial for maintainability and onboarding new developers.

8.  **Testing**: Implement unit tests for individual functions (e.g., `update_conversation_history`, `get_model_info`) and integration tests for the overall chat flow. This ensures that changes do not introduce regressions.

9.  **Dependency Management**: Pin exact versions of dependencies in `requirements.txt` (e.g., `openai==1.8.5`) to ensure consistent environments across different deployments.

10. **User Interface Enhancements**: While the current UI is functional, consider adding features like:
    *   Loading indicators for API calls.
    *   Clearer visual feedback for errors or successful operations.
    *   More sophisticated chat history management (e.g., search, delete individual messages).

By implementing these suggestions, you can significantly improve the robustness, scalability, and maintainability of your QnA Chatbot application.

import gradio as gr
import os
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize client with API key from environment
initial_api_key = os.getenv("OPENAI_API_KEY")

# Global variables
current_api_key = initial_api_key
current_model = "gpt-4o-mini"
client = None
conversation_history = []
system_prompt = "You are a helpful, creative, and intelligent AI assistant. You provide accurate, detailed, and engaging responses while being friendly and professional."

# Initialize client if API key is available
if initial_api_key:
    client = AsyncOpenAI(api_key=initial_api_key)

# Predefined system prompts
# Constants for system prompt choices
DEFAULT_ASSISTANT = "Default Assistant"
CUSTOM_PROMPT = "Custom"

SYSTEM_PROMPTS = {
    DEFAULT_ASSISTANT: "You are a helpful, creative, and intelligent AI assistant. You provide accurate, detailed, and engaging responses while being friendly and professional.",
    "Creative Writer": "You are a creative writing assistant. Help users with storytelling, creative writing, poetry, and imaginative content. Be expressive and inspiring.",
    "Code Expert": "You are a programming expert. Provide clear, well-commented code solutions, explain programming concepts, and help debug issues. Focus on best practices and clean code.",
    "Academic Tutor": "You are an academic tutor. Explain complex concepts clearly, provide step-by-step solutions, and help students understand difficult topics across various subjects.",
    "Business Analyst": "You are a business consultant. Provide strategic insights, analyze market trends, suggest business solutions, and help with professional decision-making.",
    "Health & Wellness": "You are a health and wellness advisor. Provide general health information, wellness tips, and lifestyle advice. Always remind users to consult healthcare professionals for medical issues.",
    "Travel Guide": "You are a travel expert. Provide destination recommendations, travel tips, cultural insights, and help plan memorable trips around the world.",
    "Tech Support": "You are a technical support specialist. Help troubleshoot technology issues, explain technical concepts simply, and provide step-by-step solutions.",
}

async def get_openai_response_stream(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=2000):
    """Get streaming response from OpenAI API"""
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield full_response
    except Exception as e:
        yield f"Error: {str(e)}"

def update_conversation_history(user_msg, assistant_msg, system_msg):
    """Update the global conversation history"""
    global conversation_history
    
    # Add system message if it's the first message or changed
    if not conversation_history or conversation_history[0]["content"] != system_msg:
        conversation_history = [{"role": "system", "content": system_msg}]
    
    # Add user and assistant messages
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history.append({"role": "assistant", "content": assistant_msg})
    
    # Keep only last 20 messages (plus system prompt) to manage context length
    if len(conversation_history) > 41:  # 1 system + 40 messages
        conversation_history = conversation_history[:1] + conversation_history[-40:]

async def chat_response_stream(message, history, api_key, model, temperature, max_tokens, system_prompt_choice, custom_system_prompt):
    """Handle streaming chat response"""
    global client, current_api_key, current_model, system_prompt
    
    # Update API key if provided and different
    if api_key and api_key.strip() != current_api_key:
        current_api_key = api_key.strip()
        if current_api_key:
            client = AsyncOpenAI(api_key=current_api_key)
        else:
            client = None
    elif not api_key and current_api_key:
        current_api_key = None
        client = None

    # Check if client exists
    if client is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "🔑 Please provide your OpenAI API key to start the conversation."})
        yield history, ""
    
    # Update model
    current_model = model
    
    # Update system prompt
    if system_prompt_choice == "Custom" and custom_system_prompt.strip():
        system_prompt = custom_system_prompt.strip()
    else:
        system_prompt = SYSTEM_PROMPTS.get(system_prompt_choice, SYSTEM_PROMPTS["Default Assistant"])
    
    if not message.strip():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please enter a message to continue our conversation."})
        yield history, ""
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "🤔 Thinking..."})
    
    # Prepare messages for API
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[:-1]:  # Exclude the "Thinking..." message
        messages.append(msg)
    
    # Get streaming response
    full_response = ""
    try:
        async for partial_response in get_openai_response_stream(messages, current_model, temperature, max_tokens):
            full_response = partial_response
            history[-1]["content"] = full_response
            yield history, ""
        
        # Update conversation history for context
        update_conversation_history(message, full_response, system_prompt)
        
    except Exception as e:
        history[-1]["content"] = f"❌ Error: {str(e)}"
        yield history, ""

def clear_chat():
    """Clear the chat history and conversation memory"""
    global conversation_history
    conversation_history = []
    return []

def export_conversation(history):
    """Export conversation as JSON"""
    if not history:
        return None
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "model": current_model,
        "system_prompt": system_prompt,
        "conversation": history
    }
    
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return filename

def get_status_info():
    """Get current status information"""
    status = f"🟢 **Active** | Model: {current_model} | Messages: {len(conversation_history)} | Time: {datetime.now().strftime('%H:%M:%S')}"
    return status

def get_model_info(model):
    """Get information about the selected model"""
    model_info = {
        "gpt-3.5-turbo": "⚡ **GPT-3.5 Turbo** - Fast and efficient for most tasks. Cost-effective choice.",
        "gpt-4": "🧠 **GPT-4** - Most capable model with superior reasoning. Higher cost but better quality.",
        "gpt-4-turbo": "🚀 **GPT-4 Turbo** - Latest GPT-4 with improved performance and larger context window.",
        "gpt-4o": "✨ **GPT-4o** - Optimized for conversation with multimodal capabilities.",
        "gpt-4o-mini": "💫 **GPT-4o Mini** - Lightweight version of GPT-4o. Great balance of speed and capability."
    }
    return model_info.get(model, "📋 Model information not available.")

# Enhanced CSS with ChatGPT-like styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #10a37f;
    --primary-hover: #0d8d6c;
    --secondary-color: #f7f7f8;
    --accent-color: #0066cc;
    --success-color: #10a37f;
    --warning-color: #ff9500;
    --error-color: #ff3333;
    --background-primary: #ffffff;
    --background-secondary: #f7f7f8;
    --background-chat: #ffffff;
    --text-primary: #2d3748;
    --text-secondary: #4a5568;
    --text-light: #718096;
    --border-color: #e2e8f0;
    --shadow-primary: 0 2px 8px rgba(0, 0, 0, 0.1);
    --shadow-secondary: 0 1px 3px rgba(0, 0, 0, 0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.gradio-container {
    background: var(--background-primary);
    color: var(--text-primary);
    min-height: 100vh;
}

/* Header styling */
.header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
    color: white;
    padding: 20px;
    text-align: center;
    border-radius: 0 0 20px 20px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-primary);
}

.header h1 {
    font-size: 2.5em;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header p {
    font-size: 1.1em;
    margin: 10px 0 0 0;
    opacity: 0.9;
}

/* Settings panel styling */
.settings-panel {
    background: var(--background-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-secondary);
}

.settings-panel h3 {
    color: var(--text-primary);
    margin-top: 0;
    margin-bottom: 15px;
    font-weight: 600;
}

/* Chat container styling */
.chat-container {
    background: var(--background-chat);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: var(--shadow-secondary);
    overflow: hidden;
}

/* Status bar styling */
.status-bar {
    background: linear-gradient(135deg, var(--success-color), var(--accent-color));
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 0.9em;
    font-weight: 500;
}

/* Model info styling */
.model-info {
    background: var(--background-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    font-size: 0.9em;
}

/* Button styling */
.gr-button {
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.2s ease;
    cursor: pointer;
}

.gr-button:hover {
    background: var(--primary-hover);
    transform: translateY(-1px);
}

.gr-button[variant="secondary"] {
    background: var(--warning-color);
}

.gr-button[variant="secondary"]:hover {
    background: #e6850e;
}

/* Input styling */
.gr-textbox, .gr-dropdown, .gr-slider {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    transition: border-color 0.2s ease;
}

.gr-textbox:focus, .gr-dropdown:focus {
    border-color: var(--primary-color);
    outline: none;
    box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
}

/* Chatbot styling */
.gr-chatbot {
    background: transparent;
    border: none;
    font-size: 14px;
    line-height: 1.6;
}

/* Message styling */
.message {
    padding: 15px;
    margin: 8px 0;
    border-radius: 12px;
    max-width: 85%;
    word-wrap: break-word;
}

.message.user {
    background: var(--primary-color);
    color: white;
    margin-left: auto;
    margin-right: 0;
}

.message.assistant {
    background: var(--background-secondary);
    color: var(--text-primary);
    margin-right: auto;
    margin-left: 0;
}

/* Advanced controls */
.advanced-controls {
    background: var(--background-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.advanced-controls h4 {
    margin-top: 0;
    margin-bottom: 10px;
    color: var(--text-primary);
}

/* Responsive design */
@media (max-width: 768px) {
    .header h1 {
        font-size: 2em;
    }
    
    .settings-panel {
        padding: 15px;
    }
    
    .gr-button {
        padding: 8px 16px;
        font-size: 13px;
    }
}

/* Loading animation */
@keyframes thinking {
    0%, 80%, 100% { opacity: 1; }
    40% { opacity: 0.3; }
}

.thinking {
    animation: thinking 1.5s infinite;
}

/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.3s ease-out;
}
"""

# Create the ChatGPT-like interface
with gr.Blocks(css=custom_css, title="🤖 ChatGPT-like AI Assistant", theme=gr.themes.Soft()) as demo:
    # Header
    with gr.Row(elem_classes="header"):
        gr.HTML("""
        <div>
            <h1>🤖 ChatGPT-like AI Assistant</h1>
            <p>Powered by OpenAI's GPT models with advanced conversation features</p>
        </div>
        """)
    
    with gr.Row():
        # Left column - Settings and controls
        with gr.Column(scale=1):
            with gr.Group(elem_classes="settings-panel fade-in"):
                gr.Markdown("### ⚙️ **Configuration**")
                
                # API Key
                api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    type="password",
                    value=initial_api_key,
                    placeholder="sk-...",
                    info="Your OpenAI API key for accessing GPT models"
                )
                
                # Model Selection
                model_dropdown = gr.Dropdown(
                    choices=[
                        "gpt-4o-mini",
                        "gpt-4o",
                        "gpt-4-turbo",
                        "gpt-4",
                        "gpt-3.5-turbo"
                    ],
                    value="gpt-4o-mini",
                    label="🧠 AI Model",
                    info="Choose the GPT model for your conversation"
                )
                
                # Model information display
                model_info_display = gr.Markdown(
                    get_model_info("gpt-4o-mini"),
                    elem_classes="model-info"
                )
                
                # System Prompt Selection
                system_prompt_dropdown = gr.Dropdown(
                    choices=list(SYSTEM_PROMPTS.keys()) + ["Custom"],
                    value="Default Assistant",
                    label="🎭 Assistant Personality",
                    info="Choose how the AI should behave"
                )
                
                # Custom System Prompt
                custom_system_prompt = gr.Textbox(
                    label="✏️ Custom System Prompt",
                    placeholder="Enter your custom system prompt here...",
                    lines=3,
                    visible=False,
                    info="Define custom behavior for the AI"
                )
                
            # Advanced Controls
            with gr.Group(elem_classes="advanced-controls fade-in"):
                gr.Markdown("### 🎛️ **Advanced Settings**")
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature",
                    info="Higher values = more creative, lower = more focused"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=2000,
                    step=100,
                    label="📝 Max Tokens",
                    info="Maximum length of the AI response"
                )
                
            # Control Buttons
            with gr.Group(elem_classes="settings-panel fade-in"):
                gr.Markdown("### 🎮 **Controls**")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    export_btn = gr.Button("📥 Export", variant="primary")
                
                # Status Display
                status_display = gr.Markdown(
                    get_status_info(),
                    elem_classes="status-bar"
                )
                
            # Tips and Info
            with gr.Group(elem_classes="settings-panel fade-in"):
                gr.Markdown("""
                ### 💡 **Tips**
                - **Temperature**: 0.3-0.7 for focused responses, 0.7-1.2 for creative writing
                - **Max Tokens**: Higher values allow longer responses but cost more
                - **System Prompts**: Try different personalities for varied conversation styles
                - **Memory**: The AI remembers your conversation context automatically
                """)
        
        # Right column - Chat interface
        with gr.Column(scale=2):
            with gr.Group(elem_classes="chat-container fade-in"):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 **Conversation**",
                    height=600,
                    show_copy_button=True,
                    show_label=True,
                    container=True,
                    type="messages",
                    avatar_images=("https://cdn-icons-png.flaticon.com/512/847/847969.png", 
                                 "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")
                )
                
                # Message input
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Type your message here... (Shift+Enter for new line)",
                        lines=2,
                        max_lines=6,
                        scale=4,
                        container=False,
                        autofocus=True
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)
    
    # Welcome message
    with gr.Group(elem_classes="settings-panel fade-in"):
        gr.Markdown("""
        ## 🚀 **Welcome to Your AI Assistant!**
        
        This ChatGPT-like interface offers advanced features for natural conversation:
        
        ### ✨ **Key Features**
        - **🧠 Multiple AI Models**: Choose from GPT-3.5, GPT-4, and GPT-4o variants
        - **🎭 Personality Modes**: Pre-configured system prompts for different use cases
        - **💾 Conversation Memory**: Maintains context throughout your session
        - **🎛️ Advanced Controls**: Fine-tune temperature and response length
        - **📥 Export Conversations**: Save your chats as JSON files
        - **⚡ Streaming Responses**: Real-time response generation
        
        ### 🎯 **Getting Started**
        1. **Add your OpenAI API key** (required for functionality)
        2. **Choose your preferred model** (GPT-4o-mini recommended for speed)
        3. **Select an assistant personality** or create your own
        4. **Start chatting** - the AI will remember your conversation!
        
        *Built By ❤️ Mahfujul Karim*
        """)
    
    # Event handlers
    async def submit_message(message, history, api_key, model, temperature, max_tokens, system_prompt_choice, custom_system_prompt):
        """Handle message submission"""
        if not message.strip():
            yield history, ""
        
        # Use async generator for streaming
        async for x in chat_response_stream(message, history, api_key, model, temperature, max_tokens, system_prompt_choice, custom_system_prompt):
            yield x
    
    # In on_system_prompt_change function:
    def on_system_prompt_change(choice):
        if choice == CUSTOM_PROMPT:
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    def on_model_change(model):
        """Handle model selection change"""
        return get_model_info(model), get_status_info()
    
    def refresh_status():
        """Refresh status display"""
        return get_status_info()
    
    # Connect events
    send_btn.click(
        submit_message,
        inputs=[msg, chatbot, api_key_input, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_dropdown, custom_system_prompt],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        submit_message,
        inputs=[msg, chatbot, api_key_input, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_dropdown, custom_system_prompt],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot]
    )
    
    system_prompt_dropdown.change(
        on_system_prompt_change,
        inputs=[system_prompt_dropdown],
        outputs=[custom_system_prompt]
    )
    
    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown],
        outputs=[model_info_display, status_display]
    )
    
    # Auto-refresh status
    demo.load(refresh_status, outputs=[status_display])

# Launch the application
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        show_api=False,
        quiet=False,
        ssl_verify=False
    )
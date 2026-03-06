"""
Gradio UI for the MCP Weather Assistant.
Provides a chat interface for interacting with the MCP server.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("./env.dev")

import asyncio
import gradio as gr
from gradio.components.chatbot import ChatMessage

from utils.mcp_client_wrapper import MCPClientWrapper
from config import GRADIO_HOST, GRADIO_PORT


def gradio_interface():
    """Create and configure the Gradio interface."""
    
    async def submit_message(message, chat_history, upload_media):
        """Handle message submission and stream responses."""
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
        
        # Immediately append the user's message to chat history
        chat_history += [{"role": "user", "content": message}]
        
        # Yield the updated chat history and clear the input box immediately
        yield chat_history, "", None, None
        
        # Create a new client for each request to avoid state issues
        client = MCPClientWrapper()
        
        # Now stream the assistant's response and update the chat
        async for updated_history, textbox, image_data, audio_data in client.process_message(message, chat_history, upload_media):
            yield updated_history, textbox, image_data, audio_data
    
    with gr.Blocks(title="MCP Weather Client") as demo:
        gr.Markdown("# MCP Weather Assistant")
        gr.Markdown("Connect to your MCP weather server and chat with the assistant")
        
        # State variables to store uploaded images and audio
        image_state = gr.State(None)
        audio_state = gr.State(None)
        
        with gr.Row(equal_height=True):
            # Left side - main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[], 
                    height=500,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about weather or alerts (e.g., What's the weather in New York?)",
                        scale=5
                    )
                    clear_btn = gr.Button(
                        "Clear Chat", 
                        scale=1
                    )
            
            # Right side - media upload and display
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Media")
                
                # Image upload section with icon
                with gr.Group():
                    upload_file = gr.File(
                        label="Upload Image/Audio file",
                        file_count="single",
                        file_types=["audio", "image"]    
                    )
                    process_btn = gr.Button("Process")
                
                gr.Markdown("### 🎧 Audio Player")
                
                # Audio player/output section
                output_audio = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    show_label=False
                )
                
                gr.Markdown("### 🖼️ Image Display")
                
                # Image display section
                display_image = gr.Image(
                    label="Generated Image",
                    show_label=False
                )
            
            # Connect the components
            msg.submit(
                submit_message, 
                inputs=[msg, chatbot, upload_file], 
                outputs=[chatbot, msg, display_image, output_audio]
            )
            
            clear_btn.click(
                lambda: ([], None, None, None), 
                None, 
                [chatbot, upload_file, display_image, output_audio]
            )
            
    return demo


async def main():
    """Main entry point for the Gradio app."""
    # Test connection to MCP server
    client = MCPClientWrapper()
    await client.check_connection()
    await client._connect()
    
    # Launch Gradio interface
    interface = gradio_interface()
    interface.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        debug=True
    )


if __name__ == "__main__":
    asyncio.run(main())


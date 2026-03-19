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

async def gradio_interface():
    """Create and configure the Gradio interface."""
    # Test connection to MCP server
    client = MCPClientWrapper()
    # await client.check_connection()
    # await client._connect()
    
    def toggle_media_active(file):
        if file is not None:
            return gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False, value=None), True, False, gr.update(interactive=False)
        return gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), False, False, gr.update(interactive=True)

    def toggle_doc_active(file):
        if file is not None:
            return gr.update(visible=False, interactive=False, value=None), gr.update(visible=True, interactive=True), False, True, gr.update(interactive=True)
        return gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), False, False, gr.update(interactive=True)

    def toggle_rag_ui(rag_enabled):
        # Toggle document upload visibility/interactivity based on RAG checkbox
        doc_visible = rag_enabled
        return gr.update(visible=doc_visible, interactive=doc_visible), gr.update(visible=doc_visible), doc_visible

    def upload_document(document_file):
        client.process_document_file(document_file)
    
    async def submit_message(message, chat_history, upload_media, rag_enabled):
        """Handle message submission and stream responses."""
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
        
        # Immediately append the user's message to chat history
        chat_history += [{"role": "user", "content": message}]
        
        # Yield the updated chat history and clear the input box immediately
        yield chat_history, ""
        
        # Now stream the assistant's response and update the chat
        async for updated_history, textbox in client.process_message(message, chat_history, upload_media, rag_enabled):
            yield updated_history, textbox
    
    with gr.Blocks(title="MCP Weather Client") as demo:
        gr.Markdown("# MCP Weather Assistant")
        gr.Markdown("Connect to your MCP weather server and chat with the assistant")
        
        # State variables to store uploaded images and audio
        image_state = gr.State(None)
        audio_state = gr.State(None)
        media_active = gr.State(value=False)
        doc_active = gr.State(value=False)
        rag_enabled_state = gr.State(value=True)  # RAG enabled by default
        
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
                # gr.Markdown("### 📁 Upload Media")
                
                # Image upload section with icon
                with gr.Group():
                    process_btn = gr.Button("Ingest", visible=True)
                    upload_file = gr.File(
                        label="Upload Image/Audio file",
                        file_count="single",
                        file_types=["audio", "image"],
                        visible=True
                    )
                    upload_document_file = gr.File(
                        label="Upload Document file (RAG)",
                        file_count="single",
                        file_types=["text", ".pdf", ".doc", ".docx"],
                        visible=True
                    )
                    upload_document_file.change(
                        toggle_doc_active,
                        inputs=[upload_document_file],
                        outputs=[upload_file, upload_document_file, media_active, doc_active, process_btn]
                    )
                    upload_file.change(
                        toggle_media_active,
                        inputs=[upload_file],
                        outputs=[upload_file, upload_document_file, media_active, doc_active, process_btn]
                    )
                    
                # RAG Checkbox Toggle
                rag_checkbox = gr.Checkbox(
                    label="Enable RAG",
                    value=False,
                    info="Toggle to enable/disable document upload and RAG retrieval"
                )
                
                rag_checkbox.change(
                    toggle_rag_ui,
                    inputs=[rag_checkbox],
                    outputs=[upload_document_file, process_btn, doc_active]
                )
                
                # gr.Markdown("### Document Upload (RAG Controlled)")
                
        process_btn.click(
            upload_document,
            inputs=[upload_document_file]
        )
        
        # Connect the components
        msg.submit(
            submit_message,
            inputs=[msg, chatbot, upload_file, rag_checkbox], 
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            lambda: ([], "", None, None, None, False, False), 
            None, 
            [chatbot, msg, upload_file, upload_document_file, media_active, doc_active]
        )
        
    return demo


async def main():
    """Main entry point for the Gradio app."""
    # Launch Gradio interface
    interface: gr.Blocks = await gradio_interface()
    interface.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        debug=True
    )


if __name__ == "__main__":
    asyncio.run(main())


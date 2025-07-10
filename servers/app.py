import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("./env.dev")

import asyncio
import gradio as gr
from utils.mcp_client_wrapper import MCPClientWrapper

client = MCPClientWrapper()

def gradio_interface():
    def process_message_with_image(message, chat_history, img):
        # Process the user's text message and get image data
        updated_history, _, image_data = client.process_message(message, chat_history)
        
        # If an image was uploaded, add it to the chat history
        if img is not None:
            updated_history.append((None, (img,)))
        
        # Return updated history, clear the input textbox, and update display_image with image_data
        return updated_history, "", image_data

    async def submit_message(message, chat_history, img):
        # Immediately append the user's message to chat history
        if chat_history is None:
            chat_history = []
        chat_history = chat_history + [{"role": "user", "content": message}]
        
        # Yield the updated chat history and clear the input box immediately
        yield chat_history, "", img
        
        # Now stream the assistant's response and update the chat
        async for updated_history, textbox, image_data in client.process_message(message, chat_history, img):
            yield updated_history, textbox, image_data
    
    with gr.Blocks(title="MCP Weather Client") as demo:
        gr.Markdown("# MCP Weather Assistant")
        gr.Markdown("Connect to your MCP weather server and chat with the assistant")
        
        # State variables to store uploaded images
        image_state = gr.State(None)
        
        with gr.Row(equal_height=True):
            # Left side - main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[], 
                    height=500,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about weather or alerts (e.g., What's the weather in New York?)",
                        scale=4
                    )
                    clear_btn = gr.Button("Clear Chat", scale=1)
            
            # Right side - image upload and display
            with gr.Column(scale=1):
                upload_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                display_image = gr.Image(
                    label="Generated Image",
                    height=300
                )
                
            # Connect the components
            msg.submit(
                submit_message, 
                inputs=[msg, chatbot, upload_image], 
                outputs=[chatbot, msg, upload_image]
            )
            
            clear_btn.click(
                lambda: ([], None), 
                None, 
                [chatbot, upload_image]
            )
            
            # Optional: Function to handle image generation and display
            def update_display_image(image_data):
                return image_data
            
            # Add a listener for when images are processed/generated
            # This is a placeholder - you'll need to connect it to your actual image generation process
            upload_image.change(
                update_display_image,
                inputs=[upload_image],
                outputs=[display_image]
            )
            
        return demo

async def main():
    client = MCPClientWrapper()
    
    await client.check_connection()
    await client._connect()
    interface = gradio_interface()
    interface.launch(debug=True)

if __name__ == "__main__":
    asyncio.run(main())
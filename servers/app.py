import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("./env.dev")

import asyncio
from PIL import Image
import gradio as gr
from gradio.components.chatbot import ChatMessage
from utils.mcp_client_wrapper import MCPClientWrapper
from utils.utils import normalize_audios, _read_wav_from_bytes

client = MCPClientWrapper()

def gradio_interface():
    async def submit_message(message, chat_history, upload_media):
        # Immediately append the user's message to chat history
        if chat_history is None:
            chat_history = []
        chat_history += [{"role": "user", "content": message}]
        
        # Yield the updated chat history and clear the input box immediately
        yield chat_history, "", None, None
        
        # Now stream the assistant's response and update the chat
        async for updated_history, textbox, image_data, audio_data in client._process_message(message, chat_history, upload_media):
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
                    # height=60,
                    show_label=False
                )
                
                gr.Markdown("### 🖼️ Image Display")
                
                # Image display section
                display_image = gr.Image(
                    label="Generated Image",
                    # max_height=150,
                    show_label=False
                )
                
            # Connect the components
            msg.submit(
                submit_message, 
                inputs=[msg, chatbot, upload_file], 
                outputs=[chatbot, msg, upload_file, upload_file]
            )
            
            clear_btn.click(
                lambda: ([], None, None, None), 
                None, 
                [chatbot, upload_file]
            )
            
            # Debug media
            # def process_file(file):
            #     if file is None:
            #         return None, None
                
            #     # Determine file type
            #     file_path = file.name if hasattr(file, 'name') else file
                
            #     if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            #         # Process image
            #         image_data = Image.open(file)
                    
            #         return None, image_data
            #     else:
            #         try:
            #             with open(file, 'rb') as f:
            #                 audio_bytes = f.read()
            #         except:
            #             audio_bytes = file.read()
                        
            #         audio_data, sr = _read_wav_from_bytes(audio_bytes)
            #         print(audio_data, sr)
            #         return (sr, audio_data), None
                
            # process_btn.click(
            #     fn=process_file,
            #     inputs=[upload_file],
            #     outputs=[output_audio, display_image]
            # )
        return demo

async def main():
    client = MCPClientWrapper()
    
    await client.check_connection()
    await client._connect()
    interface = gradio_interface()
    interface.launch(debug=True)

if __name__ == "__main__":
    asyncio.run(main())
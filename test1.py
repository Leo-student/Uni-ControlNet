import gradio as gr

def greet(name):
    return f"Hello {name}!"

block = gr.Interface(fn=greet, inputs="text", outputs="text")

block.launch(server_name='0.0.0.0', share=True)

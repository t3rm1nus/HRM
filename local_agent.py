import gradio as gr
from openai import OpenAI

# El cliente toma API Key y API Base desde las variables de entorno
client = OpenAI()

def responder(prompt):
    response = client.chat.completions.create(
        model="gpt-5-chat-latest",  # Cambia al modelo que prefieras
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

iface = gr.Interface(fn=responder, inputs="text", outputs="text", title="Agente DeepSeek")
iface.launch()

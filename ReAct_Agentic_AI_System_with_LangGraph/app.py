import gradio as gr
from agent import run_agent

def call(prompt):
    return run_agent(prompt)

ui = gr.Interface(
    fn=call,
    inputs=gr.Textbox(label="Ask HealthBuddy"),
    outputs=gr.Textbox(label="Response"),
    title="HealthBuddy AI Agent"
)

ui.launch()
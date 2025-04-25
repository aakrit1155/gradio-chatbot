import gradio as gr


choices = ['OPENAI ChatGPT', 'Google GenAI', 'Anthropic Claude']
gr.Dropdown(choices=choices, multiselect=False, filterable=True, label="Select AI Model", info="Select the desired AI model to use for this chat.", show_label=True, )




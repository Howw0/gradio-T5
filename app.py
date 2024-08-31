import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


sourceLangList = ["English"]
translationLangList = ["English", "French", "Romanian", "German"]

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")


def translate(text: str, sourceLang: str, translationLang: str):
    """
    Translation function
    """
    try:
        if sourceLang not in sourceLangList:
            return gr.Error(message=f"Invalid source language: {sourceLang}. Must be one of {sourceLangList}.")
        if translationLang not in translationLangList:
            return gr.Error(message=f"Invalid translation language: {translationLang}. Must be one of {translationLangList}.")
        prompt = f"translate {sourceLang} to {translationLang}: ~{text}~"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=64)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return gr.Error(message=f"An unexpected error has occured, try later.")
    return result


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Text to translate"),
        gr.Dropdown(label="Source Language", choices=sourceLangList, value="English"),
        gr.Dropdown(label="Target Language", choices=translationLangList)
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="T5 Translation",
    description="Translate text from one language to another using the T5 model."
)

demo.launch()

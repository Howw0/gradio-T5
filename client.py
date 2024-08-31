from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
		text="Your provided code ",
		sourceLang="English",
		translationLang="French",
		api_name="/predict"
)
print(result)
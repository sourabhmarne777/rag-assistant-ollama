from langchain.llms import Ollama

llm = Ollama(model="mistral")

def get_answer(prompt):
    return llm.predict(prompt)

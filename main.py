from gpt4all import GPT4All
from pathlib import Path
import os
import pandas as pd
from config import *
def createModel():
    model_path = Path(os.environ['LOCALAPPDATA']) / 'nomic.ai' / 'GPT4All'
    model = GPT4All(model_name=MODEL_NAME, model_path=model_path, device='NVIDIA GeForce RTX 3070')
    return model

def loadData():
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    data = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', header=None, names=columns)
    data = data[['text', 'polarity']]
    if SAMPLE_SIZE is not None:
        data = data.sample(n=SAMPLE_SIZE)

    return data

def chat(model, text):
    with model.chat_session(SYSTEM_PROMPT) as session:
        response = session.generate(PROMPT_TEMPLATE.format(text))
    print(response)
    return response


def predict(model, data):
    results = []
    for index, row in data.iterrows():
        text = row['text']
        print("Query ", index, ": " + text)
        response = chat(model, text)
        results.append((text, row['polarity'], response))
    return results

def evaluate_predictions(predictions):
    correct = 0
    for _, polarity, response in predictions:
        print("Model Prediction: " + response + "Truth: " + str(polarity))
        if response == str(polarity):
            correct += 1
    return correct / len(predictions)




if __name__ == '__main__':
    model = createModel()
    data = loadData()
    predictions = predict(model, data)
    accuracy = evaluate_predictions(predictions)
    print(f"Model Accuracy: {accuracy:.2%}")



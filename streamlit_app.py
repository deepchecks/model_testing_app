import streamlit as st
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification

api = HfApi()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Show title and description
st.title("🤖 Model Testing Platform")
st.write(
    "This app allows you to test different text classification models. "
    "Select a model, enter some text, and the app will classify it."
)


# Load available models from the 'models/' folder
def load_models():
    models = api.list_models(author="Nadav-Deepchecks")
    model_names = [model.modelId for model in models]
    return model_names


# Create a dropdown to select the model
st.write("Loading a model may take up to 2 minutes")
model_name = st.selectbox("Select a model", load_models())

# Load the selected text classification model and tokenizer
def load_selected_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


# Create a text input field for the user
tokenizer, model = load_selected_model(model_name)
text = st.text_area("Enter some text to classify:")

# Classify the text when the user clicks the "Classify" button
if st.button("Classify"):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Use the model to get the predicted label and score

    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]

    # Display the classification result
    st.write(f"The text was classified as '{predicted_label}'")

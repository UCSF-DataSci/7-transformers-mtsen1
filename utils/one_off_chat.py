# utils/one_off_chat.py

import requests
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_response(prompt, model_name="google/flan-t5-base", api_key=None):
    """
    Get a response from the model
    
    Args:
        prompt: The prompt to send to the model
        model_name: Name of the model to use
        api_key: API key for authentication (optional for some models)
        
    Returns:
        The model's response
    """
    # TODO: Implement the get_response function
    # Set up the API URL and headers
    # Create a payload with the prompt
    # Send the payload to the API
    # Extract and return the generated text from the response
    # Handle any errors that might occur
    if not hasattr(get_response, "tokenizer") or not hasattr(get_response, "model"):
        get_response.tokenizer = AutoTokenizer.from_pretrained(model_name)
        get_response.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenizer = get_response.tokenizer
    model = get_response.model

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def run_chat(model_name="google/flan-t5-base"):
    """Run an interactive chat session"""
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response from the model
        # Print the response

        response = get_response(user_input, model_name=model_name)
        print("LLM:", response)
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    # TODO: Add arguments to the parser

    parser = argparse.ArgumentParser(description="Chat with a local Hugging Face LLM")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="Model name to load locally")
    
    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments

    run_chat(args.model)
    
if __name__ == "__main__":
    main()
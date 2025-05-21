# utils/conversation.py

import requests
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_response(prompt, history=None, model_name="google/flan-t5-base", history_length=3):
    """
    Get a response from the model using conversation history
    
    Args:
        prompt: The current user prompt
        history: List of previous (prompt, response) tuples
        model_name: Name of the model to use
        api_key: API key for authentication
        history_length: Number of previous exchanges to include in context
        
    Returns:
        The model's response
    """
    # TODO: Implement the contextual response function
    # Initialize history if None
    if history is None:
        history = []

    if not hasattr(get_response, "tokenizer") or not hasattr(get_response, "model"):
        get_response.tokenizer = AutoTokenizer.from_pretrained(model_name)
        get_response.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    tokenizer = get_response.tokenizer
    model = get_response.model
        
    # TODO: Format a prompt that includes previous exchanges
    # Get a response from the API
    # Return the response
    context = ""
    for user_text, assistant_text in history[-history_length:]:
        context += f"User: {user_text}\nAssistant: {assistant_text}\n"
    context += f"User: {prompt}\nAssistant: "
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith(context):
        response = response[len(context):].strip()
    
    return response

def run_chat(model_name="google/flan-t5-base"):
    """Run an interactive chat session with context"""
    print("Welcome to the Contextual LLM Chat! Type 'exit' to quit.")
    
    # Initialize conversation history
    history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response using conversation history
        # Update history
        # Print the response
    
        response = get_response(user_input, history=history, model_name=model_name)
        print("Assistant:", response)

        history.append((user_input, response))
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM using conversation history")
    # TODO: Add arguments to the parser

    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="Model name to load locally")
    
    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments

    run_chat(args.model)
    
if __name__ == "__main__":
    main()
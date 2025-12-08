import json
import requests
from src.config import LLM_API_KEY


def call_llm_api(prompt_text: str):
    """
    Sends the prompt to the LLM via the REST API and streams the result.

    Args:
        prompt_text: The formatted prompt string for the LLM.

    Returns:
        The LLM's generated answer as a string.
    """
    # yield "This "
    # yield "is "
    # yield "a "
    # yield "mock."
    # return "Mock response"
    # Based on LLM_API.txt content
    LLM_API_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
    API_KEY = LLM_API_KEY

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": f"{API_KEY}"
    }

    # Based on LLM_API.txt content
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt_text,
        "stream": True, # Set to True to get streaming response as shown in the example
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        with requests.post(LLM_API_URL, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Process streaming response chunk by chunk
            for chunk in response.iter_content(chunk_size=None):
                try:
                    # Decode the chunk to string and split into lines
                    chunk_str = chunk.decode('utf-8')
                    json_lines = chunk_str.strip().split('\n')

                    for line in json_lines:
                        if line:
                            json_data = json.loads(line)
                            if 'response' in json_data:
                                token = json_data['response']
                                # print(token, end='') # Print token in real-time
                                yield token
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Warning: Could not decode or parse chunk: {chunk}. Error: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return "I'm sorry, I couldn't get a response from the LLM at this moment."

def call_llm_api_full(prompt_text: str):
    full_text = ""
    for token in call_llm_api(prompt_text):
        full_text += token
    return full_text
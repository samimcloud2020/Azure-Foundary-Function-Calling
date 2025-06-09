import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import requests
import urllib.parse

# Load environment variables
load_dotenv('azureopenai.env')

# Initialize client
endpoint = "https://____________________.cognitiveservices.azure.com/"
model_name = "gpt-4o-mini"  # Fixed from o4-mini
deployment = "gpt-4o-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Weather function
def get_weather(location):
    if not location:
        return {"error": "Invalid location"}
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return {"error": "Missing WEATHER_API_KEY in azureopenai.env"}
    # URL-encode the location to handle commas and spaces
    encoded_location = urllib.parse.quote(location)
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={encoded_location}&aqi=no"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise for HTTP errors
        data = response.json()
        return {
            "location": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "lat": data["location"]["lat"],
            "lon": data["location"]["lon"],
            "weather": data["current"]["condition"]["text"],
            "temp_c": data["current"]["temp_c"]
        }
    except requests.exceptions.HTTPError as e:
        return {"error": f"Weather API HTTP error: {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API request failed: {e}"}
    except (KeyError, ValueError):
        return {"error": "Invalid response from Weather API"}

# Main loop with conversation history
messages = [{"role": "system", "content": "You are a weather assistant. For any weather query (e.g., 'weather of London', 'London', or '51.5074,-0.1278'), use the get_weather function with the location (city name or coordinates). If the location is ambiguous (e.g., multiple Londons), ask for clarification (e.g., London, UK vs. London, Canada). If the user confirms a location (e.g., 'London, UK' or 'yes' to a suggested location), call get_weather immediately. Do not provide coordinates yourself."}]
while (msg := input("Message (end to stop): ")).lower() != "end":
    try:
        messages.append({"role": "user", "content": msg})
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=1000,
            model=deployment,
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Retrieve weather data for a given location (city name or coordinates)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Location name (e.g., 'London, UK') or coordinates (e.g., '51.5074,-0.1278')"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            tool_choice="auto"
        )
        choice = response.choices[0].message
        if choice.tool_calls:
            args = json.loads(choice.tool_calls[0].function.arguments)
            result = get_weather(args["location"])
            print(json.dumps(result, indent=2))
            messages.append({"role": "assistant", "content": json.dumps(result)})
        else:
            print(choice.content)
            messages.append({"role": "assistant", "content": choice.content})
    except Exception as e:
        print(f"Error: {e}. Verify AZURE_OPENAI_API_KEY and WEATHER_API_KEY in azureopenai.env, deployment '{deployment}' in Azure Portal, and endpoint.")
print("Ended.")

client.close()

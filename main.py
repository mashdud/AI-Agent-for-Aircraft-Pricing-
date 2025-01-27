from dotenv import load_dotenv
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain.tools import tool
import requests
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import os
import re


# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")


def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching access token: {e}")
        raise

@tool("lookup_airport", return_direct=False)
def airport_lookup_tool(query: str) -> str:
    """
    Looks up airport codes for a given city or country.
    Example: 'France' or 'South Africa'
    """
    try:
        access_token = get_access_token()
        if not access_token:
            return "Failed to authenticate with the Amadeus API."

        url = "https://test.api.amadeus.com/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "subType": "CITY,AIRPORT",
            "keyword": query,
            "page[limit]": 5
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        locations = response.json()

        if not locations.get("data"):
            return f"No airports found for {query}"

        results = []
        for location in locations["data"]:
            code = location.get("iataCode", "")
            name = location.get("name", "")
            city = location.get("address", {}).get("cityName", "")
            country = location.get("address", {}).get("countryName", "")
            
            if code and name:
                results.append(f"{code} ({name}, {city}, {country})")

        return "\n".join(results) if results else f"No airports found for {query}"

    except Exception as e:
        return f"Error looking up airport: {e}"

@tool("search_flights", return_direct=True)
def flight_search_tool(query: str) -> str:
    """
    Searches for flights using the Amadeus API.
    The query should be a string containing origin, destination, and optional budget.
    Example: '{"origin": "CDG", "destination": "JNB", "budget": 500}'
    """
    try:
        import json
        try:
            params = json.loads(query.replace("'", '"'))
        except json.JSONDecodeError:
            return "Invalid input format. Please provide input as a JSON string."

        origin = params.get("origin", "")
        destination = params.get("destination", "")
        budget = float(params.get("budget", float('inf')))
        departure_date = params.get("departure_date", datetime.now().strftime("%Y-%m-%d"))

        if not origin or not destination:
            return "Both origin and destination are required."

        access_token = get_access_token()
        if not access_token:
            return "Failed to authenticate with the Amadeus API."

        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": 1,
            "currencyCode": "USD"
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        flight_offers = response.json()

        if not flight_offers.get("data"):
            return "No flights found for the given parameters."

        results = []
        for offer in flight_offers["data"]:
            price = float(offer["price"]["total"])
            if budget and price > budget:
                continue
                
            airline = ", ".join(offer["validatingAirlineCodes"])
            departure = offer["itineraries"][0]["segments"][0]["departure"]["at"]
            arrival = offer["itineraries"][0]["segments"][-1]["arrival"]["at"]
            
            results.append(
                f"Airline: {airline}\nPrice: ${price:.2f}\n"
                f"Departure: {departure}\nArrival: {arrival}\n"
            )

        if not results:
            return f"No flights found under ${budget}" if budget else "No flights found."
            
        return "\n".join(results)

    except Exception as e:
        return f"An error occurred while searching for flights: {e}"



def preprocess_query(query):
    """
    Extract origin, destination, and budget from the user query using regex.
    Handles various formats of location specification.
    """
    # Enhanced patterns to catch more variations
    origin = re.search(r"(?:from|in)\s+([a-zA-Z\s]+?)(?:\s+to|\s*\?|$)", query, re.IGNORECASE)
    destination = re.search(r"(?:to|for.*?in)\s+([a-zA-Z\s]+?)(?:\s+not|\s+please|\s*\?|$)", query, re.IGNORECASE)
    budget = re.search(r"(?:not more|under|less than|maximum)\s*(?:than)?\s*(\d+)(?:\s*dollar|\s*usd)?", query, re.IGNORECASE)
    
    return {
        "origin": origin.group(1).strip().title() if origin else None,
        "destination": destination.group(1).strip().title() if destination else None,
        "budget": float(budget.group(1)) if budget else None,
    }

def validate_query(details):
    """
    Validate the extracted query details and return an error message if required fields are missing.
    """
    missing_fields = []
    if not details.get("origin"):
        missing_fields.append("origin (e.g., 'from France')")
    if not details.get("destination"):
        missing_fields.append("destination (e.g., 'to South Africa')")
    if missing_fields:
        return f"Missing details: Please specify {', '.join(missing_fields)}."
    return None



# Create the prompt template with preprocessed information
template = """
You are a helpful flight search assistant. Your goal is to help users find flights within their budget.
The query has been preprocessed and these details were extracted:
Origin: {origin}
Destination: {destination}
Budget: ${budget} USD

Follow these steps:
1. First look up the airport codes for {origin}
2. Then look up the airport codes for {destination}
3. Use the most appropriate airport codes to search for flights within the budget

Use this format:
Thought: I'll start by finding airports in {origin}
Action: lookup_airport
Action Input: "{origin}"
Observation: [airport codes and details]
Thought: Now I'll find airports in {destination}
Action: lookup_airport
Action Input: "{destination}"
Observation: [airport codes and details]
Thought: I'll search for flights using the most suitable airport codes
Action: search_flights
Action Input: {{"origin": "XXX", "destination": "YYY", "budget": {budget}}}
Observation: [flight search results]
Thought: [analyze the results]
Final Answer: [provide a clear response with flight options]

Available Tools:
{tools}

Tool Names:
{tool_names}

User Query:
{query}
{agent_scratchpad}
"""

def create_flight_search_agent():
    """
    Create and return a configured flight search agent.
    """
    llm = OpenAI()
    tools = [airport_lookup_tool, flight_search_tool]
    
    # Create the prompt with all required variables
    prompt = PromptTemplate(
        template=template,
        input_variables=["tools", "query","tool_names", "agent_scratchpad", "origin", "destination", "budget"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_flight_query(user_query):
    """
    Process a flight search query from start to finish.
    """
    # Preprocess and validate the query
    query_details = preprocess_query(user_query)
    validation_error = validate_query(query_details)
    
    if validation_error:
        return validation_error
    
    # Create agent and execute search
    agent_executor = create_flight_search_agent()
    
    # Set a default budget if none specified
    if query_details["budget"] is None:
        query_details["budget"] = float('inf')
    
    # Execute the search with preprocessed details
    try:
        result = agent_executor.invoke({
            "query": user_query,
            "origin": query_details["origin"],
            "destination": query_details["destination"],
            "budget": query_details["budget"],
            "agent_scratchpad": ""
        })
        return result
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

# Example usage
if __name__ == "__main__":
    test_queries = [
        "i am looking for a flight to south africa not more 1000dollar please i am from france?",
        

    ]
    
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = process_flight_query(query)
        print(f"Result: {result}")
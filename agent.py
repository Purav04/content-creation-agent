import autogen
from autogen import UserProxyAgent, ChatCompletion, GroupChatManager, AssistantAgent
from serpapi import GoogleSearch


import os
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")

topic = "content generation using AI"

config_list = [
    {
        "model": "llama-3.3-70b-versatile",
        "api_key": groq_api_key,
        "base_url": "https://api.groq.com/openai/v1"
    }
]

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config={
        "last_n_messages": 2,
        "use_docker": False
    }
)

researcher = AssistantAgent(
    name="researcher",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    }
)

def serpapi_research(query, num_results=5):
    params = {
        "q": query,
        "api_key": serp_api_key,
        "num": num_results
    }

    search_result = GoogleSearch(params)
    result = search_result.get_dict()

    search_results = []
    if "organic_results" in result:
        for result in result["organic_results"]:
            title = result.get("title", "No Title")
            link = result.get("link", "No Link")
            snippet = result.get("snippet", "No Snippet")
            search_results.append({"title":title, "link":link, "snippet":snippet})
    else:
        search_results = "No search result found"

    return search_results


@user_proxy.register_for_execution()
def research(search_query):
    search_results = serpapi_research(search_query)
    if isinstance(search_query, list):
        filtered_result = ""
        for i,result in enumerate(search_results):
            filtered_result += f"Result {i+1}:\n"
            formatted_results += f"Title: {result['title']}\n"
            formatted_results += f"Link: {result['link']}\n"
            formatted_results += f"Snippet: {result['snippet']}\n\n"
            formatted_results += f"{'-'*50}"
        return filtered_result
    else:
        return search_results

user_proxy.initiate_chat(
    researcher,
    message=topic
)
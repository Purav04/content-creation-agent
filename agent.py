import autogen
from autogen import UserProxyAgent, GroupChatManager, GroupChat, AssistantAgent

from bs4 import BeautifulSoup

import os, json, requests
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")


config_list = [
    {
        "model": "llama-3.3-70b-versatile",
        "api_key": groq_api_key,
        "base_url": "https://api.groq.com/openai/v1"
    }
]


def serpapi_search(query, num_results=5):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serp_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

def scrap_data(url:str):

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        return text
    else:
        print(f"HTTP request error for url: {url}")
        return ""

def researcher_function(query):
    llm_config_for_research = {
            "functions": [
            {
                "name": "serpapi_search",
                "description": "google search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "number of result get from google search"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "scrap_data",
                "description": "Get data by scraping data from url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "website url for scrapping"
                        }
                    },
                    "required": ["url"]
                }
            }
        ],
        "config_list": config_list
    }


    researcher = AssistantAgent(
        name="researcher",
        system_message="Research about a given query, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; Add TERMINATE to the end of the research report;",
        llm_config=llm_config_for_research
    )

    user_proxy1 = UserProxyAgent(
        name="User_Proxy",
        code_execution_config={
            "last_n_message":2,
            "work_dir": "coding",
            "use_docker": False
        },
        human_input_mode="TERMINATE",
        function_map={
            "search": serpapi_search,
            "scrape": scrap_data
        }
    )

    user_proxy1.initiate_chat(researcher, message=query)
    user_proxy1.stop_reply_at_receive(researcher)
    user_proxy1.send("Give me the research report that just generated again, return ONLY the report & reference links", researcher)

    return user_proxy1.last_message()["content"]

def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message="You are a senior editor of an AI blogger, you will define the structure of a short blog post based on material provided by the researcher, and give it to the writer to write the blog post",
        llm_config={"config_list": config_list},
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message="You are a professional AI blogger who is writing a blog post about AI, you will write a short blog post based on the structured provided by the editor, and feedback from reviewer; After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="You are a world class hash tech blog content critic, you will review & critic the written blog and provide feedback to writer.After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    user_proxy2 = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        human_input_mode="TERMINATE",
    )

    groupchat = GroupChat(
        agents=[user_proxy2, editor, writer, reviewer],
        messages=[],
        max_round=10
    )
    manager = GroupChatManager(groupchat=groupchat)

    user_proxy2.initiate_chat(
        manager, message=f"write a blog regarding {topic} based on material: {research_material}"
    )

    user_proxy2.stop_reply_at_receive(manager)
    user_proxy2.send("Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

    return user_proxy2.last_message()["content"]

llm_config_overall= {
    "functions": [
        {
            "name": "researcher_function",
            "description": "research about a given topic, return the research material including reference links",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "name of topic on which research will happen"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "write_content",
            "description": "write a content based on research material and topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "research_material": {
                        "type": "string",
                        "description": "research material of a given topic, including reference links when available"
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic of the content"
                    }
                },
                "required": ["research_material", "topic"]
            }
        }
    ],
    "config_list": config_list
}


writing_assistant = AssistantAgent(
    name="writing_assistant",
    system_message="You are a writing assistant, you can use research function to collect latest information about a given topic, and then use write_content function to write a very well written content; Reply TERMINATE when your task is done",
    llm_config=llm_config_overall
)

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="TERMINATE",
    function_map={
        "write_content": write_content,
        "research": researcher_function
    },
    code_execution_config={
            "use_docker": False}
)

user_proxy.initiate_chat(writing_assistant, message="write a blog about autogen multi AI agent framework")







from phi.agent import Agent
from phi.tools import Toolkit
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app

from bs4 import BeautifulSoup

import os, json, requests, pprint
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")
phidata_api_key = os.getenv("PHIDATA_API_KEY")


@dataclass
class SearchScrap(Toolkit):
    def __init__(self, name = "toolkit"):
        super().__init__(name)
        self.register(self.search_scrap)
        # self.query = query
        # self.num_results = num_results

    def search_scrap(self, query, num_results:int = 5):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': serp_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code == 200:
            data = []
            for json_data in response.json()["organic"]:
                url = json_data["link"]

                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    text = soup.get_text()
                    if len(text) > 800:
                        data.append(text[:800])
                    else:
                        data.append(text)
                    
                else:
                    print(f"HTTP request error for url: {url}")
            
            return str(data)
        else:
            print(f"status code: {response.status_code}")
            return ""


    
query = "Phidata multi AI agent framework"

research_agent = Agent(
    name="Research agent",
    role=f"Research about a given {query}, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; keep num_result between 0 to 10",
    tools = [SearchScrap()],
    show_tool_calls=True,
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)
)

# output = research_agent.run()

writer_agent = Agent(
        name="Writer Agent",
        role=f"You are a professional AI blogger who is writing a blog post about AI, you will write a short blog post based on the structured provided research_agent, and feedback from reviewer;",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)
    )

reviewer_Agent = Agent(
    name="Reviewer Agent",
    system_message="You are a world class hash tech blog content critic, you will review & critic the written blog and provide feedback to writer_agent.After 2 rounds of content iteration, add TERMINATE to the end of the message",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)
)

# agent_team = Agent(
#     team=[research_agent, writer_agent, reviewer_Agent],
#     model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
#     show_tool_calls=True,
#     markdown=True,
# )

# agent_team.print_response(query, stream=False)

app = Playground(agents=[research_agent, writer_agent, reviewer_Agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground_phidata:app", reload=True)

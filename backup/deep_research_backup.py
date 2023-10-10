import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from memory_local.summary_buffer_local import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import re

from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. Tool for search

business_insider_cookies = {
    'permutive-id': '964e78f0-168e-4d75-a392-af6831f15886',
    '_hjSessionUser_968386': 'eyJpZCI6IjcyMWU2Y2NhLTNkZmItNWZhZS1iMmVmLTk5N2M4MjZhNGFhNCIsImNyZWF0ZWQiOjE2Nzc2MTA1OTM4NTYsImV4aXN0aW5nIjpmYWxzZX0=',
    '_awl': '2.1685992313.5-7fb5f18636f02188d809ceef687302c0-6763652d75732d7765737431-0',
    '_ABTest_smart-banner': 'control',
    'universal_id': '3c1df332-0a3f-4301-8bf4-8913f905b3b1',
    'anonymous_id': 'd2109dca-eeb1-4cd0-ac9b-ccb7346ffbc2',
    '__adblocker': 'false',
    'fenrir_anonymous_id': 'd2109dca-eeb1-4cd0-ac9b-ccb7346ffbc2',
    '_pcid': '%7B%22browserId%22%3A%22lkp36q26be060tig%22%7D',
    '__pnahc': '0',
    'crossdomain_id_set': '1',
    '__pat': '-14400000',
    'xbc': '%7Bkpcd%7DChBsa3AzNnEyNmJlMDYwdGlnEgpCNnNJWEtMaWluGjxLVGZsb2dubThxeE1keFo4TWJnWk9CYTVWb2Vxd1BMbmpRck5sejRreDlDVWJlWjUyOXNJNGIwMGdqYkggAA',
    '_pc_seen': 'A1',
    '_pcus': 'eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVHM6ODBkM2MzMmMwNGJmMjJiNWIzMTNkM2ZlNzI1YWMwODEwYmYwMmYxOTpub19zY29yZSIsIkxUYzozMGQzY2JiMDdjMWFjMWRmMThjMWVmMzYwMTcyMDEwNWEzMzZmNmQ2Om5vX3Njb3JlIiwiTFRyZWc6MTY3MGY5NmEwYTNlYjRiNzRhOWE3YzkzZGM5ZjQ5NjI1MzkxMDU0Yjpub19zY29yZSJdfX19',
    'IR': 'F',
    'insider_uid': '906ffdeb-2f2d-46c1-83c5-c0c14c78ebce',
    '_pc_bipf': '1',
    'hasPrime': '1',
    'fenrir_insider_uid': '906ffdeb-2f2d-46c1-83c5-c0c14c78ebce',
    '_pctx': '%7Bu%7DN4IgrgzgpgThIC5QAUByBJAJgNwIoE8AvMABwHNsZMB7RUEmKAMwEsAPREAIxZZABoQAF3wkonAGoANEAF9ZgyLADKQgIZDInRmRYQhsKJgEgILA1k4BGKwGYAHAFZnANgBMt249sAWFwHZ-H385IA',
    'osano_consentmanager_uuid': 'fdb89480-f0c1-4cf3-96d9-76f59c17bbd4',
    'osano_consentmanager': 'V1N530JpeljwcU-AJt9X3khahSFIRA0jp6GafHkArQEjMHlPzQhU3yq_LC1XUw5CP43vYukcXNIl_Uq67-sxQgvH1qJzkNBj8I8_if6hd-Auwxk5Q_dfpPKs3-rg7lWuh7-0hg6yfI-J_kgv9lb0HxZQaLlFyGn1AavsF5TnftRXZ_noHVoBeNy--U9taP0Ov8SklIRYZBAWMN3vmM6jMLZxFQvGJg4wWIgGtSlwg6hiR0zZZAUQFYSDrvGt9RQUChAvJkAf2jNA5GlM2Zequbtm-IiyvxU8QzP9biELq54IqoJMiEQM7Cl-9Zvor3LazqgfK88GFbAU0ohWKRhqfGCJcO27cI8JjhdzXM7Qjho1u9OLJhwxps9kYX5WoFgKYNplZ694KKzLatbnvZthItPaVHBTpjoxmbfuN-1rgY7T7qUhNajOSIaGuibuiaHeW-KAk3E1yyUxGVFqfgNA1nXK9O4AhxQo0IdTzV8xwtQIY7yH6KL4PB271N5ep9QNrx3osG96lhhmFULmOVd6MObfS6pf6nAwWnT4e9JVHGm6UzcFdioZ1pZFSHWz8rdmC3e2TJ692gAqL4vs_kZZM1o6tWCvUqNa0AAYsATaM7MGpuBBpT7ZDLNdyxO5qpEf-6FtUSzFQ8LxRDwXnEGSvuqbGcy8_Zga8LbKActSAzLJqLW9J49K7gxktw57DS0xoVhCyIdtD_G9OEMHrhP7vUm3my4KuOr6PZ_mU63qkMiDMTgB8v0_-eWTGIqhnG3Po3Q80ex1I5ZS5iDK6xhbkXd2tdrxpLwcc2fle2agSuOEekLCnaVzcTk5fz-Ns2q_H53--zIDMl_QwHEwJe8HTFUBziqjFY7ohA8cPBhQn1O_zaWyHC-P1JMAfZN3shpV54e81FKkBOy0d6rKRzqfbL3zeGZTVxEVqTweA49YBNt0wSx5qOL3erlK3wyCYx4Msn2FOc8cjWcaV69BwbyyA_fVQNKy2EFZceJV7Za-e6UoLjB622Acd3zCwgrwJwvPl4WO6TeMR-9ruMUIU8nx3J69lII=',
    '_ABTest_taboola-widgets-for-review': 'control',
    '__cf_bm': 'L6D58vEGyjTta8g8z_Nc2sBKYdbUhRa0iQX_k.9k5Z0-1691895283-0-AZVaF1zQySxAsdsoEiQZbHzBQ9X/hgPe4ZOLMsdFyAVidnQJCD1s6oBZ4gBhr3oh+m7oZH6QRYStagH87r/aN38=',
    'piano_limit': '1',
    '_pcid': '%7B%22browserId%22%3A%22lkp36q26be060tig%22%7D',
    '_pctx': '%7Bu%7DN4IgrgzgpgThIC5QAUByBJAJgNwIoE8AvMABwHNsZMB7RUEmKAMwEsAPREAIxZZABoQAF3wkonAGoANEAF9ZgyLADKQgIZDInRmRYQhsKJgEgILA1k4BGKwGYAHAFZnANgBMt249sAWFwHZ-H385IA',
    '__tac': '',
    '__tae': '1691895289538',
    '__tbc': '%7Bkpcd%7DChBsa3AzNnEyNmJlMDYwdGlnEgpCNnNJWEtMaWluGjxLVGZsb2dubThxeE1keFo4TWJnWk9CYTVWb2Vxd1BMbmpRck5sejRreDlDVWJlWjUyOXNJNGIwMGdqYkggAA',
    '__pat': '-14400000',
    '__pvi': 'eyJpZCI6InYtbGw4dXJxa3VvMjA5ZHJuZSIsImRvbWFpbiI6Ii5idXNpbmVzc2luc2lkZXIuY29tIiwidGltZSI6MTY5MTg5NTI5MjE4OX0%3D',
    '_pcus': 'eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVGM6MzBkM2NiYjA3YzFhYzFkZjE4YzFlZjM2MDE3MjAxMDVhMzM2ZjZkNjpub19zY29yZSIsIkxUczo4MGQzYzMyYzA0YmYyMmI1YjMxM2QzZmU3MjVhYzA4MTBiZjAyZjE5Om5vX3Njb3JlIiwiTFRyZWc6MTY3MGY5NmEwYTNlYjRiNzRhOWE3YzkzZGM5ZjQ5NjI1MzkxMDU0Yjpub19zY29yZSJdfX19',
}
business_insider_cookies_array = [{"name": key, "value": value} for key, value in business_insider_cookies.items()]

domain = ".businessinsider.com"

# Convert the dictionary to the desired format
cookies_array_with_domain = [{"name": key, "value": value, "domain": domain} 
                            for key, value in business_insider_cookies.items() if value]


business_insider_headers = {
    'authority': 'www.businessinsider.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    # 'cookie': 'permutive-id=964e78f0-168e-4d75-a392-af6831f15886; _hjSessionUser_968386=eyJpZCI6IjcyMWU2Y2NhLTNkZmItNWZhZS1iMmVmLTk5N2M4MjZhNGFhNCIsImNyZWF0ZWQiOjE2Nzc2MTA1OTM4NTYsImV4aXN0aW5nIjpmYWxzZX0=; _awl=2.1685992313.5-7fb5f18636f02188d809ceef687302c0-6763652d75732d7765737431-0; _ABTest_smart-banner=control; universal_id=3c1df332-0a3f-4301-8bf4-8913f905b3b1; anonymous_id=d2109dca-eeb1-4cd0-ac9b-ccb7346ffbc2; __adblocker=false; fenrir_anonymous_id=d2109dca-eeb1-4cd0-ac9b-ccb7346ffbc2; _pcid=%7B%22browserId%22%3A%22lkp36q26be060tig%22%7D; __pnahc=0; crossdomain_id_set=1; __pat=-14400000; xbc=%7Bkpcd%7DChBsa3AzNnEyNmJlMDYwdGlnEgpCNnNJWEtMaWluGjxLVGZsb2dubThxeE1keFo4TWJnWk9CYTVWb2Vxd1BMbmpRck5sejRreDlDVWJlWjUyOXNJNGIwMGdqYkggAA; _pc_seen=A1; _pcus=eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVHM6ODBkM2MzMmMwNGJmMjJiNWIzMTNkM2ZlNzI1YWMwODEwYmYwMmYxOTpub19zY29yZSIsIkxUYzozMGQzY2JiMDdjMWFjMWRmMThjMWVmMzYwMTcyMDEwNWEzMzZmNmQ2Om5vX3Njb3JlIiwiTFRyZWc6MTY3MGY5NmEwYTNlYjRiNzRhOWE3YzkzZGM5ZjQ5NjI1MzkxMDU0Yjpub19zY29yZSJdfX19; IR=F; insider_uid=906ffdeb-2f2d-46c1-83c5-c0c14c78ebce; _pc_bipf=1; hasPrime=1; fenrir_insider_uid=906ffdeb-2f2d-46c1-83c5-c0c14c78ebce; _pctx=%7Bu%7DN4IgrgzgpgThIC5QAUByBJAJgNwIoE8AvMABwHNsZMB7RUEmKAMwEsAPREAIxZZABoQAF3wkonAGoANEAF9ZgyLADKQgIZDInRmRYQhsKJgEgILA1k4BGKwGYAHAFZnANgBMt249sAWFwHZ-H385IA; osano_consentmanager_uuid=fdb89480-f0c1-4cf3-96d9-76f59c17bbd4; osano_consentmanager=V1N530JpeljwcU-AJt9X3khahSFIRA0jp6GafHkArQEjMHlPzQhU3yq_LC1XUw5CP43vYukcXNIl_Uq67-sxQgvH1qJzkNBj8I8_if6hd-Auwxk5Q_dfpPKs3-rg7lWuh7-0hg6yfI-J_kgv9lb0HxZQaLlFyGn1AavsF5TnftRXZ_noHVoBeNy--U9taP0Ov8SklIRYZBAWMN3vmM6jMLZxFQvGJg4wWIgGtSlwg6hiR0zZZAUQFYSDrvGt9RQUChAvJkAf2jNA5GlM2Zequbtm-IiyvxU8QzP9biELq54IqoJMiEQM7Cl-9Zvor3LazqgfK88GFbAU0ohWKRhqfGCJcO27cI8JjhdzXM7Qjho1u9OLJhwxps9kYX5WoFgKYNplZ694KKzLatbnvZthItPaVHBTpjoxmbfuN-1rgY7T7qUhNajOSIaGuibuiaHeW-KAk3E1yyUxGVFqfgNA1nXK9O4AhxQo0IdTzV8xwtQIY7yH6KL4PB271N5ep9QNrx3osG96lhhmFULmOVd6MObfS6pf6nAwWnT4e9JVHGm6UzcFdioZ1pZFSHWz8rdmC3e2TJ692gAqL4vs_kZZM1o6tWCvUqNa0AAYsATaM7MGpuBBpT7ZDLNdyxO5qpEf-6FtUSzFQ8LxRDwXnEGSvuqbGcy8_Zga8LbKActSAzLJqLW9J49K7gxktw57DS0xoVhCyIdtD_G9OEMHrhP7vUm3my4KuOr6PZ_mU63qkMiDMTgB8v0_-eWTGIqhnG3Po3Q80ex1I5ZS5iDK6xhbkXd2tdrxpLwcc2fle2agSuOEekLCnaVzcTk5fz-Ns2q_H53--zIDMl_QwHEwJe8HTFUBziqjFY7ohA8cPBhQn1O_zaWyHC-P1JMAfZN3shpV54e81FKkBOy0d6rKRzqfbL3zeGZTVxEVqTweA49YBNt0wSx5qOL3erlK3wyCYx4Msn2FOc8cjWcaV69BwbyyA_fVQNKy2EFZceJV7Za-e6UoLjB622Acd3zCwgrwJwvPl4WO6TeMR-9ruMUIU8nx3J69lII=; _ABTest_taboola-widgets-for-review=control; __cf_bm=L6D58vEGyjTta8g8z_Nc2sBKYdbUhRa0iQX_k.9k5Z0-1691895283-0-AZVaF1zQySxAsdsoEiQZbHzBQ9X/hgPe4ZOLMsdFyAVidnQJCD1s6oBZ4gBhr3oh+m7oZH6QRYStagH87r/aN38=; piano_limit=1; _pcid=%7B%22browserId%22%3A%22lkp36q26be060tig%22%7D; _pctx=%7Bu%7DN4IgrgzgpgThIC5QAUByBJAJgNwIoE8AvMABwHNsZMB7RUEmKAMwEsAPREAIxZZABoQAF3wkonAGoANEAF9ZgyLADKQgIZDInRmRYQhsKJgEgILA1k4BGKwGYAHAFZnANgBMt249sAWFwHZ-H385IA; __tac=; __tae=1691895289538; __tbc=%7Bkpcd%7DChBsa3AzNnEyNmJlMDYwdGlnEgpCNnNJWEtMaWluGjxLVGZsb2dubThxeE1keFo4TWJnWk9CYTVWb2Vxd1BMbmpRck5sejRreDlDVWJlWjUyOXNJNGIwMGdqYkggAA; __pat=-14400000; __pvi=eyJpZCI6InYtbGw4dXJxa3VvMjA5ZHJuZSIsImRvbWFpbiI6Ii5idXNpbmVzc2luc2lkZXIuY29tIiwidGltZSI6MTY5MTg5NTI5MjE4OX0%3D; _pcus=eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJMVGM6MzBkM2NiYjA3YzFhYzFkZjE4YzFlZjM2MDE3MjAxMDVhMzM2ZjZkNjpub19zY29yZSIsIkxUczo4MGQzYzMyYzA0YmYyMmI1YjMxM2QzZmU3MjVhYzA4MTBiZjAyZjE5Om5vX3Njb3JlIiwiTFRyZWc6MTY3MGY5NmEwYTNlYjRiNzRhOWE3YzkzZGM5ZjQ5NjI1MzkxMDU0Yjpub19zY29yZSJdfX19',
    'if-modified-since': 'Sun, 13 Aug 2023 02:54:33 GMT',
    'if-none-match': 'W/"65906-eMxyXRYN7OrgZeY2yf2cc/7Zw+M"',
    'referer': 'https://www.google.com/',
    'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
}



def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.text)
    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }
    print('URL---:'+url)
    if "businessinsider" in url:
        data["cookies"] = cookies_array_with_domain
       # headers.update(business_insider_headers)
    # Convert Python object to JSON string
    data_json = json.dumps(data)
    

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    # print(response.content)


    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup.find_all("script"):
            script.decompose()
        for style in soup.find_all("style"):
            style.decompose()
        for tag in soup.find_all("li", class_="menu-item"):
            tag.extract()            
        text = soup.get_text()
        text = re.sub(r'\n+', '\n', text)

        print("CONTENTTTTTT:", text)

        if len(text) > 3000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
    ),
    ScrapeWebsiteTool()
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            Please write a comprehensive research report afte the research to show case relevant information you found. 
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles,you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ You should only use organic search results
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=3,
)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":fire:")

    st.header("AI research agent :fire:")
    query = st.text_input("Research goal")
    if query:
        st.write("Doing research for ", query)
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        result = agent({"input": query}, callbacks=[st_cb])

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
# app = FastAPI()


# class Query(BaseModel):
#     query: str


# @app.post("/")
# def researchAgent(query: Query):
#     query = query.query
#     content = agent({"input": query})
#     actual_content = content['output']
#     return actual_content

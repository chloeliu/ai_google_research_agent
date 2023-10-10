import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import requests
import json
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain import PromptTemplate, LLMChain

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


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

    print(response.text)

    return response.text

def summary_snippet(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    prompt = PromptTemplate(
    input_variables=["objective","text"],
    template="""
    You are a world class researcher, who can will answer question base on the search results. 
    Your goal is to write a answer given the search results based on the search query(goal) so that user can quickly get their answers without clicking into every single links. 
    '''Please make sure you complete the objective above with the following rules:
            1/ If there are url of relevant links & articles, summarize the snippet for content to answer the objective
            2/ You should not make things up, you should only write facts & data that you have gathered
            3/ You should use in-text citation style. use a reference number which is clickable link to the source article. '''
        search objective is:
        ```{objective}```
        search results is:
        ```{text}```
        """,)
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({"text":content,"objective":objective})

    return output



# Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI research agent", page_icon="🪄 ")

    st.header("AI research agent 🪄 ")
    query = st.text_input("Research goal")
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    if query:
        st.write("Search for ", query)
        data_str=search(query)
        data = json.loads(data_str)
        organic_titles = [item['title'] for item in data['organic']]
        organic_links = [item['link'] for item in data['organic']]
        with st.expander("Search Results", expanded=False):
            for title, link in zip(organic_titles, organic_links):
                st.markdown(f"[{title}]({link})")        

        organic_results = data.get('organic', [])
        result=summary_snippet(query,str(organic_results))
        # result = agent({"input": query}, callbacks=[st_cb])

        st.info(result)


if __name__ == '__main__':
    main()
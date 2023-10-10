from typing import List, Dict
from pydantic import Field
import uuid

from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from langchain.agents.agent import BaseSingleActionAgent,BaseMultiActionAgent
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)


from langchain.utilities.asyncio import asyncio_timeout

from langchain.utils.input import get_color_mapping


from langchain.agents.agent import AgentExecutor
# from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

from structured_chat_local.base import StructuredChatAgent
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
# from langchain.memory import ConversationSummaryBufferMemory
from memory_local.summary_buffer_local import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent, Tool

# from langchain_experimental.plan_and_execute.executors.base import ChainExecutor
from executor.base_local import ChainExecutor
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key=OPENAI_API_KEY)
memory = ConversationSummaryBufferMemory(
    memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=500)



# print_status_handler = PrintStatusHandler(print_status)

# HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

# Current objective: {current_step}

# {agent_scratchpad}"""

# TASK_PREFIX = """{objective}

# """
# 

HUMAN_MESSAGE_TEMPLATE = """
Rule for the task

```1/ Scraped previous search results if the url labeld "Scraped"
   2/ Do not repeat the same action twice. Always try new actions(new parameters if the results of previous step is not satisfying.
   3/ Call next action use the format Action:```$JSON_BLOB```.
```
Research History: ```{chat_history}```
Current objective: ```{current_step}```
notes: ```{agent_scratchpad}```
"""
#1/ You should do enough research so that you can provide an accurate answer to the objective 

TASK_PREFIX = """{objective}

"""

def load_agent_executor(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    verbose: bool = False,
    include_task_in_prompt: bool = False,
) -> ChainExecutor:
    """
    Load an agent executor.

    Args:
        llm: BaseLanguageModel
        tools: List[BaseTool]
        verbose: bool. Defaults to False.
        include_task_in_prompt: bool. Defaults to False.

    Returns:
        ChainExecutor
    
    """
    input_variables = ["chat_history","current_step", "agent_scratchpad"]

    template = HUMAN_MESSAGE_TEMPLATE

    if include_task_in_prompt:
        agent_kwargs['input_variables'].append("objective")
        template = TASK_PREFIX + template
    agent = StructuredChatAgent.from_llm_and_tools(
        llm,
        tools,
        human_message_template=template,
        input_variables=input_variables,
        memory=memory
    )
    # agent = OpenAIFunctionsAgent.from_llm_and_tools(
    #     llm,
    #     tools,
    #     # human_message_template=template,
    #     # input_variables=input_variables,
    #     # extra_prompt_messages=[MessagesPlaceholder(variable_name="memory")],
    #     system_message=system_message,
    #     # memory=memory
    # )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose,max_iterations=5
        , memory=memory
        ,return_intermediate_steps=True
    )
    return ChainExecutor(chain=agent_executor)

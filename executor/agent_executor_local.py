from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from structured_chat_local.base import StructuredChatAgent
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from memory_local.summary_buffer_local import ConversationSummaryBufferMemory
from executor.base_local import ChainExecutor
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose,max_iterations=5
        , memory=memory
        ,return_intermediate_steps=True
    )
    return ChainExecutor(chain=agent_executor)

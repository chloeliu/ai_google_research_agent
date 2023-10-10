# flake8: noqa
# and provide as much information as you can gather
PREFIX = """Respond to the human as accurately as possible. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer

Thought: Action to take next
...(Never make up information, always cite your sources.)
...(NEVER repeat the same Action and action inputs even if the observation is not what you expected.)

Action: 
```
$JSON_BLOB
```

Observation: action result

Thought: Action to take next
...(NEVER repeat the same Action and action inputs)
...(NEVER repeat the same Action and action inputs)

Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "answer the request carefully and cite all reference sources to back up your research;"
}}}}
```"""
SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. 
Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

# and subsequent steps 
#-- Summarize and Extract relevant information.
#-- Scrape relevant url for additional content.
#... (NEVER repeat Thought/Action/Observation)
#... (NEVER repeat previous Thought/Action/Observation)
#...(Summary and Extract all relevant information from the Observation.)

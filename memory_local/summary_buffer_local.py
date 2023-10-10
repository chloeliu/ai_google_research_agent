from typing import Any, Dict, List, Tuple

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin
from langchain.pydantic_v1 import root_validator
from langchain.schema.messages import BaseMessage, get_buffer_string


class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    """Buffer with summarizer for storing conversation memory."""
    def _get_input_output(
            self, inputs: Dict[str, Any], outputs: Dict[str, str]
        ) -> Tuple[str, str]:
            if self.input_key is None:
                prompt_input_key = 'current_step'
            else:
                prompt_input_key = 'current_step'
            if self.output_key is None:
                if len(outputs) != 1:
                    raise ValueError(f"One output key expected, got {outputs.keys()}")
                output_key = list(outputs.keys())[0]
            else:
                output_key = self.output_key
            return inputs[prompt_input_key], outputs[output_key]
        

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer = self.buffer
        if self.moving_summary_buffer != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.prune()

    def prune(self) -> None:
        """Prune buffer if it exceeds max token limit"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        outputs_single={'output': ''}
        """Save cont
        ext from this conversation to buffer."""
        print('HERE INSIDE CHAT MEMORY')
        # print('--- inputs ---')
        # print(inputs)
        # print('-'*20)
        # print('--- outputs ---')
        # # pattern = r'StepResponse\(response="([^"]+)"'
        # # match = re.search(pattern, str(outputs['output']))
        # print('is output a string?')
        if isinstance(outputs['output'], str)==True:
            # print('True')
            outputs_single['output']=outputs['output']
        # print(outputs)
        try:
            input_str, output_str = self._get_input_output(inputs, outputs_single)
        except Exception as e:
            print('ERROR HERE in Exceptions')
            print(e)
            input_str=''
            output_str=''

        # print("actual content")
        # print(str(input_str))
        # print(type(output_str))
        # print("actual content output")
        # print(output_str)
        # print('------------------------------------')
        self.chat_memory.add_user_message(str(input_str))
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.moving_summary_buffer = ""
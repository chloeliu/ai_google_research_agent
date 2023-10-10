from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain

# from langchain_experimental.plan_and_execute.executors.base import BaseExecutor
from executor.base_local import BaseExecutor  
# from langchain_experimental.plan_and_execute.planners.base import BasePlanner
from planner.base_local import BasePlanner
from langchain_experimental.plan_and_execute.schema import (
    BaseStepContainer,
    ListStepContainer,
)
from langchain.pydantic_v1 import Field
from langchain.memory import ConversationBufferMemory
import streamlit as st

class PlanAndExecute(Chain):
    """Plan and execute a chain of steps."""

    planner: BasePlanner
    """The planner to use."""
    executor: BaseExecutor
    """The executor to use."""
    step_container: BaseStepContainer = Field(default_factory=ListStepContainer)
    """The step container to use."""
    input_key: str = "input"
    output_key: str = "output"
    MAX_DEPTH = 3  # Maximum allowed recursion depth
    depth = 0  # Current recursion depth

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        if self.depth > self.MAX_DEPTH:
            print("Warning: Maximum recursion depth reached.")
            return {self.output_key: "Recursion limit reached."}
        plan = self.planner.plan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if run_manager:
            run_manager.on_text(str(plan), verbose=self.verbose)
        step_counter=0
        for step in plan.steps:            
            step_counter=step_counter+1
            if st.session_state.stop_research:
                st.warning("Research process halted by the user.")
                st.session_state.search_triggered = False
                st.session_state.stop_research = False
                st.stop() 
                break
            # print(step)
            # for prev_step, _ in self.step_container.steps:
            #     print(prev_step)            
            if step in [s[0] for s in self.step_container.steps]:
                continue  # Skip the step or handle differently
            _new_inputs = {
                # "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}
            response= self.executor.step(
                new_inputs,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            print('HERE in STEPS response')
            print(self.step_container)
            if run_manager:
                run_manager.on_text(
                    f"*****\n\nStep: {step.value}", verbose=self.verbose
                )
                run_manager.on_text(
                    f"\n\nResponse: {response.response}", verbose=self.verbose
                )
            self.step_container.add_step(step, response)
        print('------HERE in ALL STEPS response-------')
        print(self.step_container)
        st.session_state.step_list=False
        print('---------------------------------------')
        return {self.output_key: self.step_container}
        # return {self.output_key: self.step_container.get_final_response()}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        plan = await self.planner.aplan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if run_manager:
            await run_manager.on_text(str(plan), verbose=self.verbose)
        for step in plan.steps:
            _new_inputs = {
                # "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}
            response = await self.executor.astep(
                new_inputs,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            if run_manager:
                await run_manager.on_text(
                    f"*****\n\nStep: {step.value}", verbose=self.verbose
                )
                await run_manager.on_text(
                    f"\n\nResponse: {response.response}", verbose=self.verbose
                )
            self.step_container.add_step(step, response)
        return {self.output_key: str(self.step_container)}
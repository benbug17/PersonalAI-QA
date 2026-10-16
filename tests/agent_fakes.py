"""Test doubles for the agent stack: a scripted chat model that replays canned AIMessages."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Replays `responses` in order (the last one repeats forever)."""
    responses: list
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[min(self.index, len(self.responses) - 1)].model_copy(deep=True)
        # Fresh message and unique tool-call ids every time. Replaying the same object makes LangGraph's
        # message reducer treat each reply as an *update* to the previous one, so a loop would never grow.
        message.id = None
        for call in message.tool_calls:
            call["id"] = f"{call['id']}-{self.index}"
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def final(text, input_tokens=10, output_tokens=5):
    return AIMessage(content=text, usage_metadata={
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})

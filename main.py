from typing import cast

from research_agent import ResearchAgent
from research_agent.message_types import AIMessage, HumanMessage


def _extract_latest_assistant_response(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content.strip()
        if getattr(message, "role", None) == "assistant" and message.content:
            return message.content.strip()
    if messages and messages[-1].content:
        return messages[-1].content.strip()
    return ""


def main():
    agent = ResearchAgent()
    transcript: list[HumanMessage | AIMessage] = []

    print("Enter your research question. Type 'exit' or 'quit' to finish.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting conversation.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        transcript.append(HumanMessage(content=user_input))
        result = agent.chat(transcript)
        transcript = cast(list[HumanMessage | AIMessage], list(result.messages))

        assistant_reply = _extract_latest_assistant_response(result.messages)
        if assistant_reply:
            print("\nAssistant:")
            print(assistant_reply)
        else:
            print("\nAssistant: (no response)")

        if result.sources_gathered:
            print("\nSources:")
            for source in result.sources_gathered:
                label = source.label.strip()
                value = (source.value or "").strip()
                print(f"- {label}: {value}" if value else f"- {label}")

        print("-" * 100)


if __name__ == "__main__":
    main()

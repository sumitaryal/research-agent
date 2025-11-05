from research_agent import ResearchAgent


def main():
    agent = ResearchAgent()
    result = agent.chat(
        "Summarize the AI research highlights from last week"
    )
    final_message = ""
    for message in reversed(result.messages):
        if getattr(message, "role", None) == "assistant" and message.content:
            final_message = message.content.strip()
            break

    if not final_message and result.messages:
        final_message = (result.messages[-1].content or "").strip()

    if final_message:
        print("=== Answer ===")
        print(final_message)

    if result.sources_gathered:
        print()
        print("=== Sources ===")
        for source in result.sources_gathered:
            label = source.label.strip()
            value = (source.value or "").strip()
            print(f"{label}: {value}" if value else label)


if __name__ == "__main__":
    main()

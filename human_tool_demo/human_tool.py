import textwrap
import time
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType


def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0.0)
    math_llm = OpenAI(temperature=0.0)
    tools = load_tools(
        ["human", "llm-math"],
        llm=math_llm,
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent_chain.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break

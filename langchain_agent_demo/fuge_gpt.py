import textwrap
import time

from langchain_agent_demo.data_source import FugeDataSource
from langchain_agent_demo.agent import build_agent_executor
from langchain.agents import Tool
from langchain import OpenAI


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
    ## set api token in terminal
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

    fuge_data_source = FugeDataSource(llm)
    tools = [
        Tool(
            name="查询产品名称",
            func=fuge_data_source.find_product_description,
            description="通过产品名称找到产品描述时用的工具，输入应该是产品名称",
        ),
        Tool(
            name="复歌科技公司相关信息",
            func=fuge_data_source.find_company_info,
            description="当用户询问公司相关的问题，可以通过这个工具了解相关信息",
        ),
    ]

    agent_executor = build_agent_executor(llm=llm, tools=tools)

    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break

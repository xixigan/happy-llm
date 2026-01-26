from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia, get_current_temperature

from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(
        api_key="ms-5e5cbbee-fd94-41e4-8579-bafd79eb559e",
        base_url='https://api-inference.modelscope.cn/v1',
    )

    agent = Agent(
        client=client,
        model="Qwen/Qwen3-VL-8B-Instruct",
        tools=[get_current_datetime, search_wikipedia, get_current_temperature],
    )

    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答

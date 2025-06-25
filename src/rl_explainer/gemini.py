from google import genai
from google.genai import types

from rl_explainer import config, prompts
from rl_explainer.prompts import PromptWithVideo


def chat(prompt: PromptWithVideo) -> None:
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    chat = client.chats.create(model="models/gemini-2.0-flash")

    print("Generating initial analysis...")
    response = chat.send_message_stream(
        message=[
            types.Part(file_data=types.FileData(file_uri=prompt.url)),
            types.Part(
                text=(
                    f"""
                    This is a video of an already trained reinforcement learning
                    agent playing an Atari game.
                    Now I'm going to give you some details about the game:
                    {prompt.text}
                    Based on the descripton of the environment, act as an Explainable
                    Reinforcement Learning module and explain the estrategy that the
                    agent is taking. Highlight any smart strategies the agent might
                    be following.
                    """
                )
            ),
        ]
    )

    print("Gemini: ", end="")
    for chunk in response:
        print(chunk.text, end="")
    print("\n")

    while True:
        question = input("You: ")
        response = chat.send_message_stream(question)
        for chunk in response:
            print(chunk.text, end="")
        print("\n")


if __name__ == "__main__":
    print("-----------------------------------------")
    print("RL Agent Explainer Chat Interface")
    print("-----------------------------------------")

    print("Choose a game to analyze:")
    print("1: Breakout")
    print("2: Space Invaders")
    choice = input("Your choice: ")
    print("\n")

    if choice == "1":
        chat(prompts.BREAKOUT)
    elif choice == "2":
        chat(prompts.SPACE_INVADERS)
    else:
        print("\nInvalid choice. Please enter 1 or 2.\n")

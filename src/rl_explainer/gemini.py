from rl_explainer.breakout_chat import GeminiBreakoutChat


def chat() -> None:
    chat = GeminiBreakoutChat(model="models/gemini-2.0-flash")
    chat.generate_initial_analysis()
    while True:
        chat.ask()


if __name__ == "__main__":
    chat()

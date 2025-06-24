from google import genai
from google.genai import types

from rl_explainer import config, prompts

client = genai.Client(api_key=config.GEMINI_API_KEY)

# con esto puedo mandar multiples mensajes,
# quizar mandar cada estado con su metadata, accion, reward
# chat = checkout client.chats.create(model=...)
# chat.send_message(...)
#
#
# Deberia usar few shot? O quizas va a ser muy caro?

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=["Explain how AI works in a few words"],
#     # config=types.GenerateContentConfig(
#     #     thinking_config=types.ThinkingConfig(thinking_budget=0)  # disables thinking
#     # ),
# )

# print(type(response))
# print(response)
# print(response.text)
#

chat = client.chats.create(model="models/gemini-2.0-flash")
response = chat.send_message_stream(
    message=[
        types.Part(
            file_data=types.FileData(
                file_uri="https://www.youtube.com/shorts/mh98w3auWMw"
            )  # Breakout
            # file_data=types.FileData(file_uri='https://www.youtube.com/shorts/J6_kEGPib50') # Space Invaders
        ),
        types.Part(
            text=(
                f"""
                    This is a video of a reinforcement learning agent playing an Atari game.

                    Now I'm going to give you some details about the game:

                    {prompts.BREAKOUT}

                    Based on the descripton of the environment, act as an Explainable
                    Reinforcement Learning module and explain the estrategy that the agent is taking.
                    """
            )
        ),
    ]
)
for chunk in response:
    print(chunk.text, end="")

while True:
    print("Ask any other questions:")
    question = input()
    response = chat.send_message_stream(question)
    for chunk in response:
        print(chunk.text, end="")

# response = client.models.generate_content(
#     model='models/gemini-2.0-flash',
#     contents=types.Content(
#         parts=[
#             types.Part(
#                 # file_data=types.FileData(file_uri='https://www.youtube.com/shorts/mh98w3auWMw') # Breakout
#                 file_data=types.FileData(file_uri='https://www.youtube.com/shorts/J6_kEGPib50') # Space Invaders
#             ),
#             types.Part(text=(
#                     f"""
#                     This is a video of a reinforcement learning agent playing an Atari game.

#                     Now I'm going to give you some details about the game:

#                     {prompts.SPACE_INVADERS}

#                     Based on the descripton of the environment, act as an Explainable
#                     Reinforcement Learning module and explain the estrategy that the agent is taking.
#                     """
#                 )
#             ),
#         ]
#     )
# )

# In the prompt put: Info about the enviroment, possible actions, reward function.
# Idea loca, poner un layout the los controles de Atari en los que se ilumine el boton
# que se pulsa
#
#
# probablemente habilitar chat con history. Por ejemplo una vez me da la estrategia que sigue
# el agente para Breakout, preguntar por que no juega de forma normal. O por que mete la bola
# ahi en la derecha.

print(response)
print(response.text)

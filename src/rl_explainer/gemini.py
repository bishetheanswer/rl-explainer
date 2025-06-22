from google import genai

from rl_explainer import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

# con esto puedo mandar multiples mensajes,
# quizar mandar cada estado con su metadata, accion, reward
# chat = checkout client.chats.create(model=...)
# chat.send_message(...)
#
#
# Deberia usar few shot? O quizas va a ser muy caro?

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works in a few words"],
    # config=types.GenerateContentConfig(
    #     thinking_config=types.ThinkingConfig(thinking_budget=0)  # disables thinking
    # ),
)

print(type(response))
print(response)
print(response.text)

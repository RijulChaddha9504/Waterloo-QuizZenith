from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

'''from openai import OpenAI'''


# gets API Key from environment variable OPENAI_API_KEY
client = ChatOpenAI(
  openai_api_key="<REPLACE WITH YOUR API KEY>",
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="mistralai/mistral-7b-instruct:free"
)


'''completion = client.chat.completions.create(
  model="mistralai/mistral-7b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "write a blog post in Bohol island including transportation, meals, hotels and activities.",
    },
  ],
)
print(completion.choices[0].message.content)'''

template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in raps and in Anime"),
    ("user", "{input}")
])


rhyme_chain = prompt | client | StrOutputParser()

print(rhyme_chain.invoke({"input" : "Tell me about birds!"}))
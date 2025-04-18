from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os

load_dotenv()

# Step 1: Define the prompt template
template = """
You are an expert biographer. Given a detailed description of a person, generate a clean and concise summary of their background, achievements, personality, and other notable traits.

Person Description:
{person_description}

Summary:
"""

prompt = PromptTemplate(
    input_variables=["person_description"],
    template=template
)

# Step 2: Initialize the OpenAI model (GPT-4 or GPT-3.5)
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Step 3: Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Step 4: Ask for user input and run the chain
if __name__ == "__main__":
    description = """
    Assuming leadership of the Indian National Congress in 1921, Gandhi led nationwide campaigns for easing poverty, expanding women's rights, building religious and ethnic amity, ending untouchability, and, above all, achieving swaraj or self-rule. Gandhi adopted the short dhoti woven with hand-spun yarn as a mark of identification with India's rural poor. He began to live in a self-sufficient residential community, to eat simple food, and undertake long fasts as a means of both introspection and political protest. Bringing anti-colonial nationalism to the common Indians, Gandhi led them in challenging the British-imposed salt tax with the 400 km (250 mi) Dandi Salt March in 1930 and in calling for the British to quit India in 1942. He was imprisoned many times and for many years in both South Africa and India.
Gandhi's vision of an independent India based on religious pluralism was challenged in the early 1940s by a Muslim nationalism which demanded a separate homeland for Muslims within British India. In August 1947, Britain granted independence, but the British Indian Empire was partitioned into two dominions, a Hindu-majority India and a Muslim-majority Pakistan. As many displaced Hindus, Muslims, and Sikhs made their way to their new lands, religious violence broke out, especially in the Punjab and Bengal. Abstaining from the official celebration of independence, Gandhi visited the affected areas, attempting to alleviate distress. In the months following, he undertook several hunger strikes to stop the religious violence. The last of these was begun in Delhi on 12 January 1948, when Gandhi was 78. The belief that Gandhi had been too resolute in his defence of both Pakistan and Indian Muslims spread among some Hindus in India. Among these was Nathuram Godse, a militant Hindu nationalist from Pune, western India, who assassinated Gandhi by firing three bullets into his chest at an interfaith prayer meeting in Delhi on 30 January 1948.
Gandhi's birthday, 2 October, is commemorated in India as Gandhi Jayanti, a national holiday, and worldwide as the International Day of Nonviolence. Gandhi is considered to be the Father of the Nation in post-colonial India. During India's nationalist movement and in several decades immediately after, he was also commonly called Bapu, an endearment roughly meaning "father".
    """
    summary = chain.run(person_description=description)
    print("\n--- Person Summary ---")
    print(summary)

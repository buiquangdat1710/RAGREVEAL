import pandas as pd
from langchain.schema import Document
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
file_path = "chrome_debian.json"
data = pd.read_json(file_path)
# data = data.head(500)
data = data[['code', 'label']]
comment_regex = r'(//[^\n]*|\/\*[\s\S]*?\*\/)'
newline_regex = '\n{1,}'
whitespace_regex = '\s{2,}'

def data_cleaning(inp, pat, rep):
    return re.sub(pat, rep, inp)

data['code'] = (data['code'].apply(data_cleaning, args=(comment_regex, ''))
                                      .apply(data_cleaning, args=(newline_regex, ' '))
                                      .apply(data_cleaning, args=(whitespace_regex, ' '))
                         )
length_check = np.array([len(x) for x in data['code']]) > 10000
data = data[~length_check]


# Divide 80% of data for training and 20% for testing
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
test_data = test_data.head(100)
print(len(train_data))
print(len(test_data))


code_data = train_data[['code', 'label']].to_dict(orient='records')

chunks = [Document(page_content=record['code'], metadata={'label': record['label']}) for record in code_data]

# ... existing code ...

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Inserting
import pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec

pc = pinecone.Pinecone()

# for i in pc.list_indexes().names():
#     print("Deleting all indexes ...", end = "")
#     pc.delete_index(i)
#     print("Done")

index_name = "reveal"

# if index_name not in pc.list_indexes().names():
#     print(f"Creating index {index_name}")
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1",
#         )
#     )
#     print("Done")

# vector_store = Pinecone.from_documents(
#     chunks,
#     embeddings,
#     index_name=index_name,
# )

vector_store = Pinecone.from_existing_index(
    index_name=index_name,
    embedding =embeddings
)
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory


llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature=0.01)

memory = ConversationBufferMemory(
    memory_key = 'chat_history',
    return_messages=True
)



prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages = [
        SystemMessage(content='You are an expert in security, you are very good at detecting source code vulnerabilities. Think step by step to answer the question.'),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)


K = 2
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k": K})


queries = [
    f"{code}" for code in test_data['code']
]



answers = []

for query in queries:
    retrieved_docs = retriever.get_relevant_documents(query)
    ans = "### Instruction ###\nWe curate a more robust and comprehensive real world dataset, REVEAL, by tracking the past vulnerabilities from two open-source projects: Linux Debian Kernel and Chromium (open source project of Chrome). We select these projects because: (i) these are two popular and well-maintained public projects with large evolutionary history, (ii) the two projects represent two important program domains (OS and browsers) that exhibit diverse security issues, and (iii) both the projects have plenty of publicly available vulnerability reports\n### Example ###\n"
    for i, doc in enumerate(retrieved_docs[:K], 1):
        # print(f"\nDocument {i}:")
        # print(f"Code Snippet: {doc.page_content[:500]}...")  # Print first 500 characters to keep it concise
        # print(f"Label: {doc.metadata['label']}")
        label = int(doc.metadata['label'])
        prompt = "Question: See the following code:\n" +  doc.page_content + "\nThink step by step and explain each line of code in detail. "
        if label == 1:
            prompt += "This code is vulnerable. Conclude that the above code contains a vulnerability and explain why."
        else:
            prompt += "This code is not vulnerable. Conclude that the above code does not contain vulnerabilities and explain why."
        tmp = chain.invoke({'content': prompt})["text"]
        prompt += "\nAnswer:\n" + tmp + "\n"
        ans += prompt
    answers.append(ans)


idx = 0
final_answers = []
for ans in answers:
    ans = ans + "Question: See the following code:\n" + test_data['code'].iloc[idx]  + "\nThink step by step and explain each line of code in detail.\nAnswer: "
    idx += 1
    tmp = chain.invoke({'content': ans})["text"]

    final_answers.append(tmp)

for ans in final_answers:
    print(ans)
    print("-"*50)

predic_label = []
for ans in final_answers:
    prompt = ans + "\n" + "Based on the above conclusion, answer 1 if the code contains vulnerabilities, and 0 if the code does not contain vulnerabilities. You are not allowed to provide any explanations or additional information. Only respond with 0 or 1."
    tmp = chain.invoke({'content': prompt})["text"]
    predic_label.append(tmp)
print(predic_label)
# Convert answers to binary format
predictions = [1 if '1' in answer else 0 for answer in predic_label]

# Calculate metrics
labels = test_data['label'].tolist()
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
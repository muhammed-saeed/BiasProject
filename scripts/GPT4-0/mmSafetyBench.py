import pandas as pd
import csv
from openai import OpenAI

# Step 1: Load the dataset from the CSV file
df = pd.read_csv("/Users/muhammedsaeed/Desktop/BiasProject/Dataset/MM-SafetyBench_random_examples.csv")

# Check the column names (expecting 'question' column)
print(df.columns)

# Extract the questions to process
file_vlaues = df['question'].tolist()

# Step 2: Initialize the OpenAI client for llama-8b-chat
client = OpenAI(
    api_key="vqeA6msmeCasTJP0FD1sGxZnjIGXRLU1",
    base_url="https://api.lemonfox.ai/v1",
)

# Step 3: Generate responses using llama-8b-chat
generated_file_values = []

for user_content in file_vlaues:
    print(user_content)
    completion = client.chat.completions.create(
        messages=[
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": user_content }
        ],
        model="llama-8b-chat",
    )
    generated_file_values.append(completion.choices[0].message.content)

# Step 4: Save the responses generated with llama-8b-chat
with open("MM-SafetyBenchs_generated_with_llama-8b-chat.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['question', 'generated_response'])
    for i in range(len(file_vlaues)):
        writer.writerow([file_vlaues[i], generated_file_values[i]])

# Step 5: Generate responses using llama-70b-chat
generated_file_values = []

for user_content in file_vlaues:
    print(user_content)
    completion = client.chat.completions.create(
        messages=[
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": user_content }
        ],
        model="llama-70b-chat",
    )
    generated_file_values.append(completion.choices[0].message.content)

# Step 6: Save the responses generated with llama-70b-chat
with open("MM-SafetyBenchs_generated_with_llama-70b-chat.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['question', 'generated_response'])
    for i in range(len(file_vlaues)):
        writer.writerow([file_vlaues[i], generated_file_values[i]])

# Step 7: Generate responses using gpt-4o
import openai
openai.api_key = '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'

generated_file_values = []
system_content = "You are an AI assistant"

for user_content in file_vlaues:
    print(user_content)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    generated_file_values.append(response.choices[0].message.content)

# Step 8: Save the responses generated with gpt-4o
with open("MM-SafetyBenchs_generated_with_gpt-4o.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['question', 'generated_response'])
    for i in range(len(file_vlaues)):
        writer.writerow([file_vlaues[i], generated_file_values[i]])

# Step 9: Generate responses using chatgpt-4o-latest
generated_file_values = []

for user_content in file_vlaues:
    print(user_content)
    response = openai.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    generated_file_values.append(response.choices[0].message.content)

# Step 10: Save the responses generated with chatgpt-4o-latest
with open("MM-SafetyBenchs_generated_with_chatgpt-4o-latest.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['question', 'generated_response'])
    for i in range(len(file_vlaues)):
        writer.writerow([file_vlaues[i], generated_file_values[i]])

# Import necessary libraries
import pandas as pd
import csv
from openai import OpenAI

# Step 1: Load the dataset from the specified CSV file
dataset_file = "/Users/muhammedsaeed/Desktop/BiasProject/RandomSampledBaisedInput/Toxigen_random_examples (1).csv"  # Replace this with the correct CSV path
TG = pd.read_csv(dataset_file)

# Check the dataset structure
print(TG.head())

# Assuming the dataset has 'target_group' and 'text' columns
file_values = TG['text'].tolist()
target_groups = TG['target_group'].tolist()

# Initialize the OpenAI client
client = OpenAI(
    api_key="vqeA6msmeCasTJP0FD1sGxZnjIGXRLU1",  # Replace with your actual API key
    base_url="https://api.lemonfox.ai/v1",
)

# System prompt for the OpenAI API
system_content = "You are an AI assistant"

# Step 2: Generate responses using llama3-8b
generated_file_values_8b = []

for index, row in TG.iterrows():
    user_content = row['text']
    print(f"Processing text with llama3-8b: {user_content}")
    response = client.chat.completions.create(
        model="llama3-8b",  # Use the llama3-8b model
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    generated_prompt = response.choices[0].message.content
    generated_file_values_8b.append([row['target_group'], generated_prompt])

# Step 3: Save the results for llama3-8b
output_file_8b = "/Users/muhammedsaeed/Desktop/BiasProject/ModelsResponses/test/llama3-8b_Toxigen_responses.csv"
with open(output_file_8b, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['target_group', 'generated_response'])
    for row in generated_file_values_8b:
        writer.writerow(row)

print(f"Generated responses with llama3-8b saved to {output_file_8b}")

# Step 4: Generate responses using llama3-70b
generated_file_values_70b = []

for index, row in TG.iterrows():
    user_content = row['text']
    print(f"Processing text with llama3-70b: {user_content}")
    response = client.chat.completions.create(
        model="llama3-70b",  # Use the llama3-70b model
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    generated_prompt = response.choices[0].message.content
    generated_file_values_70b.append([row['target_group'], generated_prompt])

# Step 5: Save the results for llama3-70b
output_file_70b = "/Users/muhammedsaeed/Desktop/BiasProject/ModelsResponses/test/llama3-70b_Toxigen_responses.csv"
with open(output_file_70b, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['target_group', 'generated_response'])
    for row in generated_file_values_70b:
        writer.writerow(row)

print(f"Generated responses with llama3-70b saved to {output_file_70b}")

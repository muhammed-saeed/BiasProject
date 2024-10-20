# Import necessary libraries
import pandas as pd
import csv
import openai

# Step 1: Load the dataset directly from the CSV file
dataset_file = "/Users/muhammedsaeed/Desktop/BiasProject/Dataset/MM-SafetyBench_random_examples.csv"  # Replace this with the correct CSV path
TG = pd.read_csv(dataset_file)

# Check the dataset structure
print(TG.head())

# Step 2: No need to sample or shuffle the data since it's pre-sampled
# Assuming the dataset has 'target_group' and 'text' columns, you can directly feed these to the API

# Set OpenAI API key
openai.api_key = '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'

# System prompt for the OpenAI API
system_content = "You are an AI assistant"

# Step 3: Feed the data directly to the API
generated_file_values = []
for index, row in TG.iterrows():
    user_content = row['text']
    print(f"Processing text: {user_content}")
    response = openai.chat.completions.create(
        model="gpt-4o",  # You can replace this with the appropriate model
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    generated_prompt = response.choices[0].message.content
    generated_file_values.append([row['target_group'], generated_prompt])

# Step 4: Save the results to a new CSV file
output_file = "/Users/muhammedsaeed/Desktop/BiasProject/Models/test/GPT4OToxigen_random_examples_generated.csv"  # Replace with your desired path
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['target_group', 'text'])
    for generated_value in generated_file_values:
        writer.writerow(generated_value)

print(f"Generated prompts saved to {output_file}")

# Import necessary libraries
import pandas as pd
import csv
import random
import openai

# Load dataset from your folder
dataset_path = "/Users/muhammedsaeed/Desktop/BiasProject/Dataset/Toxigen_random_examples (1).csv"
TG = pd.read_csv(dataset_path)

# Show the first few rows of the dataset to verify loading
print(TG.head())

print(f"Total Rows: {TG.shape[0]}")

# Get unique target groups
print(f"Unique Target Groups: {TG['target_group'].unique()}")
for i in range(len(TG['target_group'].unique())):
  print(TG['target_group'].unique()[i])

# Seeded random number generator function
def seeded_random(seed, min_value, max_value):
    random.seed(seed)
    return random.randint(min_value, max_value)

# Function to generate random numbers until the stop condition is met
def generate_numbers_until_stop(seed, min_value, max_value, i, stop_number, arr):
    while True:
        i += 1
        random_number = seeded_random(seed, min_value, max_value)

        arr.append(random_number)
        if(i == stop_number):
          break
        seed += 1

# Set parameters for random number generation
seed = 340
min_value = 10
max_value = TG.shape[0] - 2
random_numbers_list = []
generate_numbers_until_stop(seed, min_value, max_value, 0, max_value, random_numbers_list)

# Ensure there are no duplicate random numbers
for i in range(len(random_numbers_list)):
  for j in range(len(random_numbers_list)):
    if(i != j):
      if(random_numbers_list[i] == random_numbers_list[j]):
        random_numbers_list[j] = -1

# Verify the length of the random number list
print(f"Random Numbers List Length: {len(random_numbers_list)}")

# Initialize variables
j = 0
categories = list(TG['target_group'].unique())
print(f"Number of Categories: {len(categories)}")
categories_numbers = [0]*len(categories)
file_values = []

# Select random samples from the dataset ensuring diversity across target groups
while(len(file_values) < 60):
  if(random_numbers_list[j] == -1):
    j += 1
  else:
    index = categories.index(TG.iloc[random_numbers_list[j],1])
    if(categories_numbers[index] < 4):
      categories_numbers[index] += 1
      file_values.append([TG.iloc[random_numbers_list[j],1], TG.iloc[random_numbers_list[j],0]])
      j += 1
    else:
      j += 1

# Save selected random examples to a CSV file
output_file = "random_examples.csv"
with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['target_group','text'])
  for k in range(len(file_values)):
    writer.writerow(file_values[k])

# Initialize OpenAI API key
openai.api_key = 'your-openai-api-key'

system_content = "You are an AI assistant"

# Generate responses using GPT-4 API for each selected sample
generated_file_values = []
for i in range(len(file_values)):
  user_content = file_values[i][1]
  print(f"User Content: {user_content}")
  response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
                  {"role": "system", "content": system_content},
                  {"role": "user", "content": user_content},
                  ],
                  temperature=0.7,
                  max_tokens=4000,
          )
  generated_prompt = response.choices[0].message['content']
  generated_file_values.append([file_values[i][0], generated_prompt])

# Save the generated responses to a new CSV file
output_generated_file = "Toxigen_random_examples_generated_with_gpt-4.csv"
with open(output_generated_file, 'a', newline='', encoding='utf-8') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['target_group','generated_text'])
  for k in range(len(generated_file_values)):
    writer.writerow(generated_file_values[k])

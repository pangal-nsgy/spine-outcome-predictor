import os
import pandas as pd
from openai import OpenAI

#apikey: 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


# Load your API key from an environment variable
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

# Ensure the API key is available
if not api_key:
    raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")

# Set the OpenAI API key globally


# Function to send a prompt to the OpenAI API and return the response.
def query_openai(patient_data):
    static_prompt = "What will be the long-term outcome of the patient and what is their estimated length of stay?"
    full_prompt = f"{patient_data} {static_prompt}"

    try:
        response = client.chat.completions.create(model="gpt-4 turbo", 
                                                  messages=[
                                                      {"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": full_prompt}
                                                  ])
        print("API Response:", response)  # Print the entire response
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to read data from a CSV file.
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to process the data from the DataFrame.
def process_data(df):
    for index, row in df.iterrows():
        prompt = row['Data']
        response = query_openai(prompt)
        if response and hasattr(response, 'choices') and response.choices:
            # Correctly access the content of the message
            message_content = response.choices[0].message.content
            df.at[index, 'GPT4_Response'] = message_content
        else:
            print("No valid response received or response format is unexpected.")
            df.at[index, 'GPT4_Response'] = "Error in response"
    return df


# Function to write the DataFrame to a new CSV file.
def write_to_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

# The main function where the script execution begins.
def main():
    input_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data.csv'
    output_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data_copy.csv'

    df = read_csv(input_csv)
    processed_df = process_data(df)
    write_to_csv(processed_df, output_csv)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def upload_file(file_path):
    """Uploads a file to OpenAI and returns its ID."""
    with open(file_path, 'rb') as file:
        uploaded_file = client.files.create(file=file, purpose='assistants')
    return uploaded_file.id

def create_assistant(file_id):
    """Creates an Assistant with the uploaded file."""
    assistant = client.beta.assistants.create(
        model="gpt-4-1106-preview",
        instructions="Read each patient data row and predict the long-term outcome and estimated length of stay.",
        tools=[{"type": "code_interpreter"}],
        file_ids=[file_id]
    )
    return assistant.id

def process_data(input_csv, assistant_id):
    """Processes the data from the CSV file using the Assistant."""
    df = pd.read_csv(input_csv)
    for index, row in df.iterrows():
        # Create a thread for each row of patient data
        thread = client.beta.threads.create()

        # Add message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=row['Data']
        )

        # Run the Assistant for the current row
        run = client.beta.threads.runs.create(
            assistant_id=assistant_id,
            thread_id=thread.id
        )

        # Retrieve and store the response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        responses = [msg.content for msg in messages.data if msg.role == "assistant"]
        combined_response = ' '.join(responses)
        df.at[index, 'GPT4_Response'] = combined_response

        # Print the assistant's response for debugging
        print(f"Row {index} Response: {combined_response}")

    return df



def write_to_csv(df, output_csv):
    """Writes the DataFrame to a new CSV file."""
    df.to_csv(output_csv, index=False)

def main():
    # Load your API key from an environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
    
    client.api_key = api_key

    input_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data.csv'
    output_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data_copy.csv'


    # Upload the CSV file
    file_id = upload_file(input_csv)

    # Create an Assistant
    assistant_id = create_assistant(file_id)

    # Process data and get predictions
    processed_df = process_data(input_csv, assistant_id)

    # Write predictions to a new CSV file
    write_to_csv(processed_df, output_csv)

    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()

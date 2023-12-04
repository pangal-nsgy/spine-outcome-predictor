import json
import os
import pandas as pd
from openai import OpenAI

# Load your API key from an environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")

def query_openai(patient_data):
    # Example data for few-shot learning
    example_input = ("Patient history: 62-year-old female who presents with history of seizure disorder on Keppra, "
                     "prior ACDF in Scottsdale AZ in 2003, A-fib (ASA 81), HTN, HL, hypothyroidism, "
                     "osteoporosis (Fosamax), stroke, COPD, multiple ear surgeries (deaf in right ear "
                     "and hearing aid in left), hysterectomy, cholecystectomy, and bilateral cataracts. "
                     "Surgery: C3-5 ACDF for ASD w myelopathy prior C5-7 ACDF")
    example_output = {
        "summary": "62F hx seizures on Keppra, prior ACDF, Afib on Aspirin, HTN, COPD, strokes. Uses cane to walk.",
        "length_of_stay": "3 days",
        "discharge_location": "Home"
    }

    system_message = (f"You are a helpful assistant specialized in medical data analysis. "
                      f"Provide concise predictions in JSON format based on patient history and surgery details. "
                      f"Example input: '{example_input}' Example output: {json.dumps(example_output)}")

    user_message = f"{patient_data}\n\nPredict the estimated length of stay and the most likely discharge location."

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ])
        print("API Response:", response)
        return response.choices[0].message.content if response.choices else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def construct_prompt(patient_history, surgery):
    current_input = f"Patient history: {patient_history} Surgery: {surgery}"
    return current_input

def process_data(df):
    for index, row in df.iterrows():
        prompt = construct_prompt(row['ColumnA'], row['ColumnB'])

        response = query_openai(prompt)
        if response:
            try:
                response_json = json.loads(response)
                df.at[index, 'Summary'] = response_json.get('summary', '')
                df.at[index, 'Length_of_Stay'] = response_json.get('length_of_stay', '')
                df.at[index, 'Discharge_Location'] = response_json.get('discharge_location', '')

                if index < 5:
                    print(f"Row {index+1} - JSON Response: {response_json}")

            except json.JSONDecodeError:
                print(f"Error parsing JSON response for row {index+1}")
        else:
            print(f"No valid response for row {index+1}")

    return df

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_to_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

def main():
    input_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data.csv'
    output_csv = '/Users/dhirajpangal/Desktop/Local Code/spine-outcome-predictor/test_data_copy.csv'
    
    df = read_csv(input_csv)
    processed_df = process_data(df)
    write_to_csv(processed_df, output_csv)

if __name__ == "__main__":
    main()

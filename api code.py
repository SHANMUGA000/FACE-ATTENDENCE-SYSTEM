import requests
import json

# Define the URL of the REST API
api_url = "http://172.16.71.2:9090/api/v1/JA/get_erp_student_master"  # Replace with your API URL

# Load existing data from the JSON file, if it exists
existing_data = []

try:
    with open('data.json', 'r') as json_file:
        existing_data = json.load(json_file)
except FileNotFoundError:
    pass  # The file doesn't exist initially

# Make a request to the REST API to get new data
response = requests.get(api_url)

if response.status_code == 200:
    new_data = response.json()

    # Merge new data with existing data (assuming data is a list)
    existing_data.extend(new_data)

    # Write the merged data back to the JSON file
    with open('data.json', 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    print("Data updated and saved to data.json")
else:
    print("Failed to fetch data from the API")

# Now, existing_data contains the merged data, and data.json is updated.

# Data Collection for APIs

This repository contains scripts and tools for collecting data from various API sources, including RapidAPI, LinkedIn Data API, and other public APIs. The goal of this project is to facilitate the automatic fetching and storing of API data in a structured format (JSON), making it easy for further analysis, processing, or reporting.

## Overview

The project is designed to:

- Collect data from APIs such as LinkedIn Data API (via RapidAPI).
- Store the fetched data in JSON format.
- Handle errors and retries in case of transient issues such as timeouts or rate-limiting.
- Maintain a clean structure for easy scalability and customization.

The collected data can be used for further tasks like analysis, database population, or machine learning tasks.

---

## Prerequisites

Before running the scripts in this repository, ensure you have the following:

- **Python 3.6+** installed on your machine.
- **`requests` library**: Used to make HTTP requests to the APIs.
- **`json` module**: Built-in Python library used for handling JSON data.
- **`os` module**: Built-in Python library for file and directory manipulation.

### Install Dependencies

Install the necessary Python packages using `pip`:

```bash
pip install requests
````

---

## Data Collection Scripts

### 1. LinkedIn Data API (RapidAPI)

The `linkedin_data_api.py` script collects data from the LinkedIn Data API using RapidAPI. You can access various endpoints, including profile details, company information, and posts.

#### Available Endpoints

* **User Profile**: Fetch a LinkedIn user profile based on the username.
* **Company Details**: Get details about a company, using the company name (e.g., Google).
* **Profile Posts**: Get the posts made by a LinkedIn user.

#### API Key

To use the LinkedIn Data API, you need an API key from **RapidAPI**. You can sign up on [RapidAPI](https://rapidapi.com/) and subscribe to the **LinkedIn Data API** to get your API key.

```python
headers = {
    'x-rapidapi-host': 'linkedin-data-api.p.rapidapi.com',
    'x-rapidapi-key': 'your-rapidapi-key-here'
}
```

### Example Queries in the Script

1. **Fetch LinkedIn User Profile:**

   ```bash
   curl --request GET \
     --url 'https://linkedin-data-api.p.rapidapi.com/?username=adamselipsky' \
     --header 'x-rapidapi-host: linkedin-data-api.p.rapidapi.com' \
     --header 'x-rapidapi-key: your-rapidapi-key-here'
   ```

2. **Fetch Company Details:**

   ```bash
   curl --request GET \
     --url 'https://linkedin-data-api.p.rapidapi.com/get-company-details?username=google' \
     --header 'x-rapidapi-host: linkedin-data-api.p.rapidapi.com' \
     --header 'x-rapidapi-key: your-rapidapi-key-here'
   ```

3. **Fetch Profile Posts:**

   ```bash
   curl --request GET \
     --url 'https://linkedin-data-api.p.rapidapi.com/get-profile-posts?username=adamselipsky' \
     --header 'x-rapidapi-host: linkedin-data-api.p.rapidapi.com' \
     --header 'x-rapidapi-key: your-rapidapi-key-here'
   ```

### 2. Handling Data Fetch and Save

The script allows for fetching API data and saving it to a local file in **JSON** format. The `fetch_api_data_to_json` function:

* Accepts API URL, headers, and optional parameters.
* Makes a GET or POST request to the API.
* Handles errors like HTTP issues, connection errors, timeouts, and retries.
* Saves the successfully fetched data as a **JSON** file in the specified directory.

### Example Usage

```python
# LinkedIn API headers (Replace with your own RapidAPI Key)
headers = {
    'x-rapidapi-host': 'linkedin-data-api.p.rapidapi.com',
    'x-rapidapi-key': 'your-rapidapi-key-here'
}

# Example API URL for LinkedIn Profile Fetch
api_url = "https://linkedin-data-api.p.rapidapi.com/?username=adamselipsky"

# Output file path to store data
output_filepath = "linkedin_data/profile_adamselipsky.json"

# Fetch data and save to file
fetch_api_data_to_json(api_url, output_filepath, headers)
```

The above example will fetch the profile data of the user `adamselipsky` from LinkedIn Data API and store it in the `profile_adamselipsky.json` file.

---

## Error Handling & Logging

### Key Features:

* **Retry Mechanism**: If the API request fails due to connection issues or timeouts, the script will retry the request up to 3 times (customizable).
* **Logging**: Errors, retries, and successful fetches are logged to help with debugging and monitoring the process.
* **Timeouts and Rate-Limiting**: The script handles timeouts (set to 30 seconds by default) and gracefully manages API rate limits by retrying after a delay.

### Common Errors Handled:

* **Connection Errors**: Includes network failure or invalid URL.
* **Timeout Errors**: When the API takes too long to respond.
* **HTTP Errors**: Any 4XX or 5XX response codes.
* **JSON Decode Errors**: If the response is not valid JSON.

### Logs:

Logs are captured with different levels: `INFO`, `ERROR`, and `DEBUG`. Logs can be reviewed in the terminal or saved to a file for deeper insights.

---

## Customization

You can easily customize the script by changing:

* **API URL**: Add more API endpoints by adjusting the URL.
* **API Parameters**: Pass additional query parameters or request body data for POST requests.
* **Output Path**: Define a custom directory for saving data.
* **Retries & Delay**: Adjust the retry count or delay time between retries.

---

## Structure of Fetched Data

Once the data is fetched, it's saved in a **JSON** format. The basic structure typically looks like this:

```json
{
  "id": "12345",
  "name": "John Doe",
  "profile_url": "https://www.linkedin.com/in/johndoe/",
  "headline": "Software Engineer",
  "company": "Tech Corp",
  "location": "San Francisco, CA",
  "posts": [
    {
      "post_id": "abc123",
      "content": "Excited about my new project!",
      "date": "2025-05-20"
    },
    ...
  ]
}
```

You can customize the structure based on the API response and your needs.

---

## FAQ

### 1. **How do I get my RapidAPI Key?**

* Sign up at [RapidAPI](https://rapidapi.com/).
* Find the **LinkedIn Data API** and subscribe to it.
* You'll get an API key after subscribing, which you can use in the `x-rapidapi-key` header.

### 2. **How can I fetch data for multiple users?**

* You can modify the script to loop through a list of usernames and fetch data for each user.

```python
usernames = ['adamselipsky', 'billgates', 'sundarpichai']
for username in usernames:
    api_url = f"https://linkedin-data-api.p.rapidapi.com/?username={username}"
    output_filepath = f"linkedin_data/profile_{username}.json"
    fetch_api_data_to_json(api_url, output_filepath, headers)
```

---

## Conclusion

This project provides a simple, extensible way to collect and save data from various public APIs, particularly those accessible via RapidAPI. The script is designed for flexibility, allowing for easy integration with new APIs and additional endpoints. With enhanced error handling and logging, this approach is suitable for real-world data collection tasks.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

### Key Sections in the README:

1. **Overview**: Describes the purpose of the repository and its functionality.
2. **Prerequisites**: Lists the software and libraries needed to run the scripts.
3. **Data Collection Scripts**: Provides a detailed explanation of the main functionality and how to fetch data using LinkedIn Data API.
4. **Error Handling**: Explains how the script manages errors, retries, and logs.
5. **Customization**: Instructions on how to modify the script for different API endpoints and parameters.
6. **FAQ**: Answers common questions and provides examples for extending the script.



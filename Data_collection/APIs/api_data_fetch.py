import json
import logging
import os
from time import sleep

import requests

# setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()


def fetch_api_data_to_json(
    url, output_filepath, headers, params=None, method="GET", retries=3, delay=5
):
    """
    Fetches data from a given API endpoint and stores it in a JSON file.

    Args:
        api_url (str): The URL of the API endpoint.
        output_filepath (str): The path (including filename) where the JSON data will be saved.
        headers (dict): A dictionary of HTTP headers to send with the request.
        params (dict, optional): A dictionary of URL parameters to append to the URL. Defaults to None.
        method (str, optional): The HTTP method to use (e.g., "GET", "POST"). Defaults to "GET".
        retries (int, optional): Number of retries in case of transient failures. Defaults to 3.
        delay (int, optional): Delay (in seconds) between retries. Defaults to 5.

    Returns:
        bool: True if data was successfully fetched and saved, False otherwise.
    """
    attempt = 0
    while attempt < retries:
        try:
            logger.info(f"featching dataf rom : {url}")
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=params, timeout=30)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return False

            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            data = response.json()
            logger.info(f"Successfully fetched data from {url}")

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            # Validate JSON format and structure
            if not isinstance(data, dict):
                logger.error(f"Expected a dictionary but got {type(data)}")
                return False

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"Data successfully saved to: {output_filepath}")
            return True

        except requests.exceptions.HTTPError as http_err:
            logger.error(
                f"HTTP error occurred: {http_err} - Status code: {response.status_code}"
            )
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred during the request: {req_err}")
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from response. Error: {json_err}")
            if response.text:
                logger.debug(f"Response Text: {response.text[:200]}...")
        except IOError as io_err:
            logger.error(f"File I/O error occurred: {io_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

        # Retry logic with delay
        attempt += 1
        logger.info(f"Retrying ({attempt}/{retries}) after {delay} seconds...")
        sleep(delay)

    return False


def main():
    # LinkedIn API headers (Replace with your own RapidAPI Key)
    headers = {
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com",
        "x-rapidapi-key": "0d79fa658emsh7d8d89c6b756582p180503jsn2e6a6c7bb7dd",
    }

    # API URL and Output Filepaths for each query
    queries = [
        {
            "url": "https://linkedin-data-api.p.rapidapi.com/?username=adamselipsky",
            "output": os.path.join(
                "Data", "linkedin_data", "profile_adamselipsky.json"
            ),
        },
        {
            "url": "https://linkedin-data-api.p.rapidapi.com/get-company-details?username=google",
            "output": os.path.join("Data", "linkedin_data", "company_google.json"),
        },
        {
            "url": "https://linkedin-data-api.p.rapidapi.com/get-profile-posts?username=adamselipsky",
            "output": os.path.join("Data", "linkedin_data", "posts_adamselipsky.json"),
        },
    ]

    # Loop through each query and fetch the data
    for query in queries:
        api_url = query["url"]
        output_filepath = query["output"]

        if fetch_api_data_to_json(api_url, output_filepath, headers):
            logger.info(f"Data for {api_url} fetched and saved successfully.")
        else:
            logger.error(f"Failed to fetch data for {api_url}.")


if __name__ == "__main__":
    main()

import requests
import json
import os


def fetch_api_data_to_json(
    api_url, output_filepath, headers=None, params=None, method="GET"
):
    """
    Fetches data from a given API endpoint and stores it in a JSON file.

    Args:
        api_url (str): The URL of the API endpoint.
        output_filepath (str): The path (including filename) where the JSON data will be saved.
        headers (dict, optional): A dictionary of HTTP headers to send with the request. Defaults to None.
        params (dict, optional): A dictionary of URL parameters to append to the URL. Defaults to None.
        method (str, optional): The HTTP method to use (e.g., "GET", "POST"). Defaults to "GET".

    Returns:
        bool: True if data was successfully fetched and saved, False otherwise.
    """
    try:
        print(f"Fetching data from: {api_url}")
        if method.upper() == "GET":
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
        elif method.upper() == "POST":
            # For POST, params are typically sent in the 'data' or 'json' argument
            response = requests.post(api_url, headers=headers, json=params, timeout=30)
        else:
            print(f"Unsupported HTTP method: {method}")
            return False

        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        data = response.json()
        print(f"Successfully fetched data.")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Data successfully saved to: {output_filepath}")
        return True

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
    except json.JSONDecodeError:
        print(
            f"Failed to decode JSON from response. Response text: {response.text[:200]}..."
        )  # Log first 200 chars
    except IOError as io_err:
        print(f"File I/O error occurred: {io_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return False


if __name__ == "__main__":
    # Example Usage: Fetching public API data (JSONPlaceholder)
    sample_api_url = "https://jsonplaceholder.typicode.com/todos"
    # Storing it within the Datasets/sample_data directory, as defined in your structure
    sample_output_file = os.path.join(
        "..", "..", "Data_collection", "Datasets", "sample_data", "todos_from_api.json"
    )

    if fetch_api_data_to_json(sample_api_url, sample_output_file):
        print("Example API fetch successful.")
    else:
        print("Example API fetch failed.")

    # Example with parameters (e.g., fetching a specific todo item)
    # sample_api_url_single = "https://jsonplaceholder.typicode.com/todos"
    # sample_output_file_single = os.path.join("..", "..", "Data_collection", "Datasets", "sample_data", "todo_1_from_api.json")
    # query_params = {"id": 1}
    # if fetch_api_data_to_json(sample_api_url_single, sample_output_file_single, params=query_params):
    #     print("Example API fetch for single item successful.")
    # else:
    #     print("Example API fetch for single item failed.")

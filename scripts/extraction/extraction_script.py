
import requests
import os
import json
import argparse
import time
from json.decoder import JSONDecodeError

parser = argparse.ArgumentParser()
parser.add_argument("--twitter_account", help="Name of Twitter account.",
                    type=str)
args = parser.parse_args()

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAALOEkQEAAAAAh3UmMuBNXi0t%2BK9XbzgqDEi7Dxg%3D2PkzCBZjW5dSzSY6ExVeAfy4724udCWq2qkpQfOVKMHUIMlhkX"

search_url = "https://api.twitter.com/2/tweets/search/all"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {'query': f'from:{args.twitter_account} -is:retweet','start_time': '2023-01-01T00:00:00Z',
                'end_time': '2023-03-31T23:59:59Z', 'max_results': 500}

total_tweets = 0


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)

    if 'next_token' in response.json()['meta']:
        query_params['next_token'] = response.json()['meta']['next_token']
    elif query_params.get('next_token') is not None:
        del query_params['next_token']

    return response.json()


def main():
    while True:
        global total_tweets
        before_total = total_tweets

        json_response = connect_to_endpoint(search_url, query_params)
        total_tweets = total_tweets + json_response['meta']['result_count']

        if before_total != 0:
            with open(f'tweets/{args.twitter_account}_tweets.json', 'r+', encoding='utf-8') as outfile:
                previous_object = json.load(outfile)
                for tweet in json_response['data']:
                    previous_object['tweets'].append(tweet)
                outfile.seek(0)
                json_object = json.dumps(previous_object, indent=4, sort_keys=True, ensure_ascii=False)
                outfile.write(json_object)
                outfile.truncate() 
        else:
            with open(f'tweets/{args.twitter_account}_tweets.json', 'w', encoding='utf-8') as outfile:
                tweets_object = {
                    "tweets": json_response['data']
                }
                json_object = json.dumps(tweets_object, indent=4, sort_keys=True, ensure_ascii=False)
                outfile.write(json_object)
            
        if 'next_token' not in query_params and before_total != 0:
            break

        time.sleep(10)

    print(f'Total of {args.twitter_account} tweets:', total_tweets)

if __name__ == "__main__":
    main()
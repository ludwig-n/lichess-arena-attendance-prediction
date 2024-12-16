import time


def try_get(session, url, params=None, retries=5):
    response = None
    try:
        response = session.get(url, params=params)
        response.raise_for_status()   # raises an exception if an http error occurred
    except Exception as e:
        if retries == 0:
            raise
        print("request error, retrying:", e)
        # Wait 1 minute after HTTP 429 as per API guidelines: https://lichess.org/api#section/Introduction/Rate-limiting
        time.sleep(60 if response is not None and response.status_code == 429 else 10)
        return try_get(session, url, params=params, retries=retries - 1)
    else:
        return response

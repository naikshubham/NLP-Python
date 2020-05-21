
### Twitter API
- Search API
- Ads API
- Streaming API

#### Streaming API
- Real-time tweets : The streaming API allows us to collect a sample of tweets in read-time based on keywords, user IDs and locations.
- The streaming API has two endpoints, filter and sample.

##### Filter endpoint
- We can request data on a few 100 keywords, a few thousand usernames and 25 location ranges.

##### Sample endpoint
- Twitter will return a 1% sample of all of twitter

### Using tweepy to collect data

#### `tweepy`
- Python package for accessing streaming API. It abstracts away much of the work we need to setup a stable Twitter Streaming API connection.
- Need to setup own twitter acc and API keys for authentication.

#### SListener
- `tweepy` requires an object called `SListener` which tells it how to handle incoming data.

```python
from tweepy.streaming import StreamListener
import time

class SListener(StreamListener):
    def __init__(self, api = None):
        self.output = open('tweets_%s.json' % time.strftime('%Y%m%d-%H%M%S'), 'w')
        self.api = api or API()
        ...
```

#### tweepy authentication
- OAuthentication, the authentication protocol which the Twitter API uses, requires four tokens which we obtain from the Twitter developer site : the consumer key and consumer secret, and the access token and access token secret.

```python
from tweepy import OAuthHandler
from tweepy import API

auth = OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = API(auth)
```

#### Collecting data with tweepy

```python
from tweepy import Stream

listen = SListener(api)

stream = Stream(auth, listen)

stream.sample()  # begin collecting data
```













































import os 
import sys 
from tweepy import API 
from tweepy import OAuthHandler
import json
from tweepy import Cursor 
def get_twitter_auth(): 
  """Setup Twitter authentication. 
 
  Return: tweepy.OAuthHandler object 
  """ 
  try: 
    consumer_key = 'peTLAxjDjIhEw9jD93fttR9dI'
    consumer_secret = 'wAKXbh6ZsYfBmgh0SKnF9oxldx4CXKbkOz6lBXgt8mGRU4vGqL'
    access_token = '887623406924972032-WqTWdZQQOVMbdG9LDD7iSAkh0CBaFUk'
    access_secret = 'RvCbbRZ0FpBUE0nuI3N0YFmQUZMLcLazifSDm62fs4QA9'
  except KeyError: 
    sys.stderr.write("TWITTER_* environment variables not set\n") 
    sys.exit(1) 
  auth = OAuthHandler(consumer_key, consumer_secret) 
  auth.set_access_token(access_token, access_secret) 
  return auth 
 
def get_twitter_client(): 
  """Setup Twitter API client. 
  Return: tweepy.API object 
  """ 
  auth = get_twitter_auth() 
  client = API(auth) 
  return client 
 
if __name__ == '__main__': 
  client = get_twitter_client() 
 
  with open('home_timeline.json', 'w') as f: 
    for page in Cursor(client.home_timeline, count=200).pages(4): 
      for status in page: 
        f.write(json.dumps(status._json)+"\n")

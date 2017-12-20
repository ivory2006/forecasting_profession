#import tweepy
#from tweepy import OAuthHandler
#import json
#consumer_key = 'peTLAxjDjIhEw9jD93fttR9dI'
#consumer_secret = 'wAKXbh6ZsYfBmgh0SKnF9oxldx4CXKbkOz6lBXgt8mGRU4vGqL'
#access_token = '887623406924972032-WqTWdZQQOVMbdG9LDD7iSAkh0CBaFUk'
#access_secret = 'RvCbbRZ0FpBUE0nuI3N0YFmQUZMLcLazifSDm62fs4QA9'
 
#auth = OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)
#api = tweepy.API(auth)

#timeline
#for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
#    print(status.text)
#for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
#    print(status._json)
#for friend in tweepy.Cursor(api.friends).items():
    # Process a single status
 #   print(friend._json)
#def process_or_store(tweet):
 #   print(json.dumps(tweet))

#for tweet in tweepy.Cursor(api.user_timeline).items():
    # Process a single status
 #   process_or_store(tweet._json)
# To run this code, first edit config.py with your configuration, then:
#
# mkdir data
# python twitter_stream_download.py -q apple -d data
# 
# It will produce the list of tweets for the query "apple" 
# in the file data/stream_apple.json

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string

import json
consumer_key = 'peTLAxjDjIhEw9jD93fttR9dI'
consumer_secret = 'wAKXbh6ZsYfBmgh0SKnF9oxldx4CXKbkOz6lBXgt8mGRU4vGqL'
access_token = '887623406924972032-WqTWdZQQOVMbdG9LDD7iSAkh0CBaFUk'
access_secret = 'RvCbbRZ0FpBUE0nuI3N0YFmQUZMLcLazifSDm62fs4QA9'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

 
class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#shubing'])

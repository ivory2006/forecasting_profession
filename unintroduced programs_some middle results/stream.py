from twitter_client import get_twitter_client
import json


client = get_twitter_client()

#query = client.search("shanshubing")

query=client.user_timeline("shanshubing")
print (query)
#for result in query["statuses"]:
#    if result["geo"]:
#        text=result["text"]
#        print (text)
  

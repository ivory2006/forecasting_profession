#--author:ssb--
from twitter_client import get_twitter_client
import json
import csv
exampleFile=open('newDataSample.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)
client = get_twitter_client()
#fname="1.txt"
query = client.user_timeline(exampleList[1][0])
for tweet in query:
    #print(tweet.text.encode('utf8'))
    with open("1.txt","ab") as f:
        f.write(tweet.text.encode('utf8'))
    


  

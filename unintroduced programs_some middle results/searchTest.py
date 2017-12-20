from twitter_client import get_twitter_client
import json
import csv

exampleFile=open('sample.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)
client = get_twitter_client()


query = client.search("ChemConnector")
          
for tweet in query:
    print(tweet.text.encode('utf8'))
    
            


  

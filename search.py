from twitter_client import get_twitter_client
import json
import csv
exampleFile=open('sample.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)
client = get_twitter_client()
k=678
name=k
for i in range(677,1616,1):
    try:
        fname="{}.txt".format(name)
        query = client.search(exampleList[i][0])
        for tweet in query:
            #print(tweet.text)
            with open(fname,"a") as f:
                f.write(tweet.text)
                f.close()
        name=name+1  
        
    except:
        print('Error:Unicode Encode Error or Cannot find any message of the person in the last 7 days')
        name=name+1
        





  

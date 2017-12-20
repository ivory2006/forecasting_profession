#--author:ssb--

from twitter_client import get_twitter_client
import json
import csv
import codecs
exampleFile=open('newDataSample.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)
client = get_twitter_client()
name=1617
for i in range(1617,1618,1):
    try:
        fname="{}.txt".format(name)
        query = client.user_timeline(exampleList[i][0])
        for tweet in query:
            with open(fname,"ab") as f:
                f.write(tweet.text.encode('utf8'))
                f.close()
        name=name+1  
        
    except:
        print('Error:Unicode Encode Error or Cannot find any message of the person in the last 7 days')
        name=name+1
        






  

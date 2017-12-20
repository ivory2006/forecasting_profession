from twitter_client import get_twitter_client
import json
import csv
import codecs
from time import sleep
exampleFile=open('new200k_250.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)
client = get_twitter_client()
name=24012
for i in range(24012,24652,1):
    try:
        fname="{}.txt".format(name)
        query = client.user_timeline(exampleList[i][0])
        for tweet in query:
            with open(fname,"ab") as f:
                f.write(tweet.text.encode('utf8'))
                f.close()
        print(name)
        name=name+1        
        if i%1000==0:
            time.sleep(120)
    except:
        print('Error:Unicode Encode Error or Cannot find any message of the person in the last 7 days')
        name=name+1
    

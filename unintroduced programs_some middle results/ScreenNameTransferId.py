from twitter_client import get_twitter_client
import json
import csv
exampleFile=open('new200k_250.csv')
exampleData=csv.reader(exampleFile)
exampleList=list(exampleData)

client = get_twitter_client()
for i in range(1,26257,1):
    try:
        
        
        profile = client.get_user(screen_name=exampleList[i][0]).id_str
        print(exampleList[i][0])
        
        #print(profile)
    except:
        print('Error:Cannot find the person')

import json
import csv
from pprint import pprint
with open('new200k_250.csv','r') as csvfile1:
    reader = csv.DictReader(csvfile1)
    profession=[row['Profession'] for row in reader]
    k=0
    a=0
    b=profession[k] 
    for i in range (0,24931,1):        
        if  b!=profession[k+1]:
            profession[k]=a
            b=profession[k+1]
            k=k+1
            a=a+1
            print('change',profession[k-1])
            
        else:
            profession[k]=a
            b=profession[k+1]
            k=k+1
            print('unchange',profession[k-1])

#csvfile1.close()
with open("new200k_250_number.csv","w",encoding='utf8',newline='') as csvfile2:
    
    writer=csv.writer(csvfile2)
    z=0
    writer.writerow(["Profession"])
    for j in range (0,24931,1):
        writer.writerow([profession[z]])
        z=z+1
    


    

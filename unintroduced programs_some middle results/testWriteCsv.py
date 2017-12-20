import csv
with open('test.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row['profession'] for row in reader]
    print(column[0])

csvfile.close()
with open("test.csv","w",encoding='utf8',newline='') as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name
   
    b="bname"
    c="cname"
    writer.writerow(["profession",b,c])
    
    for i in range(0,2,1):
        d=1
        f=3
        writer.writerow([column[i],d,f])
csvfile.close()

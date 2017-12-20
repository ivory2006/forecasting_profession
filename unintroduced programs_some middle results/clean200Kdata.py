import pandas as pd

df=pd.read_csv("cleantwitter200kdata.csv",low_memory=False)
grouped=df.groupby('Profession')  #the sum of group
conl=lambda s:len(s)>250
df1 = grouped.filter(conl)
df1.to_csv ("new200k_250.csv",encoding = "utf-8")

df2=pd.read_csv("new200k_250.csv",low_memory=False)
a=df2.groupby('Profession').size()
print(a)


#i = 20000
#while len(set(df2.loc[0:i,'Profession'])) != 301:
#    i = i+1


#print(i)



    

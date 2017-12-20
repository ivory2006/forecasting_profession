import codecs
k=20999
name=k
for i in range(20999,21568,1):
    try:
        fname_input="{}.txt".format(name)
        with open(fname_input,'rb') as f:
            s=f.read()
            s1=str(s)
            e=s1.encode('gbk','ignore')
            e1=str(e)
            f.close()
        with open(fname_input,'w') as w:
            w.write(e1)
            name=name+1
            
            
    except:
        print("error:cannot find the person:",name)
        name=name+1

        


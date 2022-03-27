def func(a=10,b=20,c=30):
    print(a,b,c)

func(10,**{'c':100,'b':1})
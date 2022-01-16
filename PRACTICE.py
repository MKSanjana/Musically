h,w,h1,w1 = int(input()),int(input()),int(input()),int(input())
c,ht,w1 = 0,0,0
while ht != h1:
    if(ht-h1 == 1):
        ht = h1
        print(ht)
    else:
        ht = int(h/2)
        print(ht)
        c+=1
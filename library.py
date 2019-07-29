# a〜z
l=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# A~Z
l=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Deg
l=['NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW','N']

# Beaufort scale
l=[0.2,1.5,3.3,5.4,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6]

# keyboard
s='WBWBWWBWBWBW'
t=['Do','Do','Re','Re','Mi','Fa','Fa','So','So','La','La','Si']

# GCD
def gcd(x,y):
  while y:
    x,y=y,x%y
  return x

# LCM
def lcm(x,y):
  return x*y/gcd(x,y)

# nCr
def nCr(n,r):
  if n<r:
    return 0
  return math.factorial(n)//(math.factorial(n-r)*math.factorial(r))

# struct
class struct:
  def __init__(self,a,b,c):
    self.a=a
    self.b=b
    self.c=c

# Elastotenes's sieve
def elastotenesSieve(n):
  l1=list(range(2,n+1))
  l2=[]
  
  while True:
    if len(l1)==0:
      return l2
  
    n=l1[0]
    l2.append(n)
    for y in l1:
      if y%n==0:
        l1.remove(y)

# Factoring by trial split
def getPrimeList(n):
  l=[]
  t=int(math.sqrt(n))+1
  
  for a in range(2,t):
    while n%a==0:
      n//=a
      l.append(a)
  
  if n!=1:
    l.append(n)
  
  return l

# 10 -> n
def ten2n(a,n):
  x=a//n
  y=a%n
  if x:
    return ten2n(x,n)+str(y)
  return str(y)

## usage example of ten2n
for i in range(pow(2,n)):
  print(ten2n(i,2).zfill(n))

# Summarize count of factor within list
def summarize_list(l):
  sl=sorted(l)

  a=sl[0]
  c=1
  res=[]

  for x in sl[1:]:
    if x==a:
      c+=1
    else:
      res.append([a,c])
      a=x
      c=1
  res.append([a,c])

  return res

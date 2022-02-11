# Summarize count of factor within list -- START --
def summarizeList(l):
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
# Summarize count of factor within list --- END ---

# Range Minimum Query -- START --
class SegmentTree():
  def __init__(self,v):
    sz=len(v)
    self.sz=1
    while self.sz<sz:
      self.sz*=2
    self.dat=[10**20]*(2*self.sz-1)
    for i,x in enumerate(v):
      self.update(i,x)

  def update(self,i,x):
    i+=self.sz-1
    self.dat[i]=x
    while i>0:
      i=(i-1)//2
      self.dat[i]=min(self.dat[i*2+1],self.dat[i*2+2])

  def query(self,a,b,k=0,l=0,r=-1):
    if r<0:
      r=self.sz
    if r<=a or b<=l:
      return 10**20
    if a<=l and r<=b:
      return self.dat[k]
    vl=self.query(a,b,k*2+1,l,(l+r)//2)
    vr=self.query(a,b,k*2+2,(l+r)//2,r)
    return min(vl,vr)
# Range Minimum Query --- END ---

# Range Maximum Query -- START --
class SegmentTree():
  def __init__(self,v):
    sz=len(v)
    self.sz=1
    while self.sz<sz:
      self.sz*=2
    self.dat=[-(10**20)]*(2*self.sz-1)
    for i,x in enumerate(v):
      self.update(i,x)

  def update(self,i,x):
    i+=self.sz-1
    self.dat[i]=x
    while i>0:
      i=(i-1)//2
      self.dat[i]=max(self.dat[i*2+1],self.dat[i*2+2])

  def query(self,a,b,k=0,l=0,r=-1):
    if r<0:
      r=self.sz
    if r<=a or b<=l:
      return -(10**20)
    if a<=l and r<=b:
      return self.dat[k]
    vl=self.query(a,b,k*2+1,l,(l+r)//2)
    vr=self.query(a,b,k*2+2,(l+r)//2,r)
    return max(vl,vr)
# Range Maximum Query --- END ---

# Range Xor Query -- START --
class SegmentTree():
  def __init__(self,v):
    sz=len(v)
    self.sz=1
    while self.sz<sz:
      self.sz*=2
    self.dat=[0]*(2*self.sz-1)
    for i,x in enumerate(v):
      self.update(i,x)

  def update(self,i,x):
    i+=self.sz-1
    self.dat[i]=x
    while i>0:
      i=(i-1)//2
      self.dat[i]=self.dat[i*2+1]^self.dat[i*2+2]

  def query(self,a,b,k=0,l=0,r=-1):
    if r<0:
      r=self.sz
    if r<=a or b<=l:
      return 0
    if a<=l and r<=b:
      return self.dat[k]
    vl=self.query(a,b,k*2+1,l,(l+r)//2)
    vr=self.query(a,b,k*2+2,(l+r)//2,r)
    return vl^vr
# Range Xor Query --- END ---

# BinaryIndexedTree（BIT） -- START --
class BinaryIndexedTree:
  def __init__(self,sz):
    self.sz=sz
    self.data=[0]*(sz+1)

  def sum(self,k):
    ret=0
    while k>0:
      ret+=self.data[k]
      k-=k&(-k)
    return ret

  def add(self,k,x):
    k+=1
    while k<=self.sz:
      self.data[k]+=x
      k+=k&(-k)

## BITで転倒数計算
# def main():
#   l=LI()
#   ans=0
#   bit=BinaryIndexedTree(max(l)+1)
#   for i,a in enumerate(l):
#     ans+=i-bit.sum(a+1)
#     bit.add(a,1)
#   return ans

# BinaryIndexedTree（BIT） --- END ---

# a〜z -- START --
alf=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# a〜z --- END ---

# A~Z -- START --
alf=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# A~Z --- END ---

# dijkstra -- START --
def dijkstra(n,s,edge):
  d=[10**20]*n
  used=[False]*n
  d[s]=0
  used[s]=True
  pq=[]
  for e in edge[s]:
    heapq.heappush(pq,e)
  while pq:
    c,v=heapq.heappop(pq)
    if used[v]:
      continue
    d[v]=c
    used[v]=True
    for nc,nv in edge[v]:
      if not used[nv]:
        nd=nc+c
        if d[nv]>nd:
          heapq.heappush(pq,[nd,nv])
  return d

# How to use -- START --
# Verify: https://atcoder.jp/contests/typical90/tasks/typical90_m
# 
# edge=[[] for _ in range(n)]
# for _ in range(m):
#   x,y,cost=LI()
#   x-=1
#   y-=1
#   edge[x].append([cost,y])
#   edge[y].append([cost,x])
# st=0
# d=dijkstra(n,st,edge)
# 
# How to use --- END ---
# dijkstra --- END ---

# Warshall floyd -- START --
def warshallFloyd(d):
  n=len(d)
  for k in range(n):
    for i in range(n):
      for j in range(n):
        if i==j:
          d[i][j]=0
        else:
          d[i][j]=min(d[i][j],d[i][k]+d[k][j])
  return d
# Warshall floyd --- END ---

# palindrome check(kaibun・回文) -- START --
def isPalindrome(s):
  l1=list(s)
  l2=l1[::-1]

  if l1==l2:
    return True
  return False
# palindrome check(kaibun・回文) --- END ---

# nC2 -- START --
def nC2(n):
  return n*(n-1)//2
# nC2 --- END ---

# ceil(kiriage) -- START --
def main():
  a=1.2
  math.ceil(a)
# ceil(kiriage) --- END ---

# floor(kirisute) -- START --
def main():
  a=1.2
  math.floor(a)
# floor(kirisute) --- END ---

# bit traversal -- START --
def main():
  n=I()
  l=LI()
  for i in range(1<<n):
    for j in range(n):
      if i&(1<<j):
        print(l[j])
# bit traversal --- END ---

# Sum of arithmetic progression -- START --
def sumOfArith(n,a1,an):
  return n*(a1+an)/2
# Sum of arithmetic progression --- END ---

# heapq -- START --
def main():
  q=[]
  heapq.heappush(q,3)
  x=heapq.heappop(q)
# heapq --- END ---

# deque1 -- START --
q=collections.deque()
q.append(1)
q.popleft()
while q:
# deque1 --- END ---

# deque2 -- START --
def main():
  s=S()
  s=collections.deque(s)
  s.appendleft('x')
  s.append('y')
  s=list(s)
# deque2 --- END ---

# Deg -- START --
l=['NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW','N']
# Deg --- END ---

# day of the week -- START --
l=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# day of the week --- END ---

# GCD -- START --
def gcd(x,y):
  while y:
    x,y=y,x%y
  return x
# GCD --- END ---

# LCM -- START --
def lcm(x,y):
  return x*y//gcd(x,y)
# LCM --- END ---

# osa_k method -- START --
class Osa_k():
  def __init__(self,n):
    self.n=n
    self.min_factor=[0]*(n+1)
    self.factorList=[[] for _ in range(n+1)]
    for i in range(n+1):
      self.min_factor[i]=i

    for i in range(2,int(math.sqrt(n)+1)):
      if self.min_factor[i]==i:
        for j in range(2,n//i+1):
          if self.min_factor[i*j]>i:
            self.min_factor[i*j]=i

  def getFactor(self,m):
    if len(self.factorList[m]):
      return self.factorList[m]
    tmp_factor=[]
    while m>1:
      tmp_factor.append(self.min_factor[m])
      m//=self.min_factor[m]
    self.factorList[m]=tmp_factor
    return tmp_factor
# osa_k method --- END ---

# factorial -- START --
def factorial(n):
  ret=n
  for i in range(2,n):
    ret*=i
  return ret
# factorial --- END ---

# factorial(mod) -- START --
def factorialMod(n,m):
  ret=n
  for i in range(2,n):
    ret*=i
    ret%=m
  return ret
# factorial(mod) --- END ---

# n^p(mod m) -- START --
def powMod(n,p,m):
  if p==0:
    return 1
  if p%2==0:
    t=powMod(n,p//2,m)
    return t*t%m
  return n*powMod(n,p-1,m)%m
# n^p(mod m) --- END ---

# mod inverse -- START --
pow(a,m-2,m)
# mod inverse --- END ---

# nCr(mod) -- START --
class ComMod():
  def __init__(self,n,mod):
    self.n=n
    self.mod=mod
    self.fact=[0]*(n+1)
    self.ifact=[0]*(n+1)
    self.fact[0]=1  
    for i in range(1,n+1):
      self.fact[i]=self.fact[i-1]*i
      self.fact[i]%=self.mod
    self.ifact[n]=pow(self.fact[n],self.mod-2,self.mod)
    for i in range(n,0,-1):
      self.ifact[i-1]=self.ifact[i]*i
      self.ifact[i-1]%=self.mod
  def com(self,n,k):
    if n<0 or k<0 or n<k:
      return 0
    return self.fact[n]*(self.ifact[k]*self.ifact[n-k]%self.mod)%self.mod
# nCr(mod) --- END ---

# nCr -- START --
def nCr(n,r):
  if n<r:
    return 0
  return math.factorial(n)//(math.factorial(n-r)*math.factorial(r))
# nCr --- END ---

# nCr(2) -- START --
def cmb(n,r):
  if n-r<r: r=n-r
  if r==0: return 1
  if r==1: return n

  numerator=[n-r+k+1 for k in range(r)]
  denominator=[k+1 for k in range(r)]

  for p in range(2,r+1):
    pivot=denominator[p-1]
    if pivot>1:
      offset=(n-r)%p
      for k in range(p-1,r,p):
        numerator[k-offset]/=pivot
        denominator[k]/=pivot

  result=1
  for k in range(r):
    if numerator[k]>1:
      result*=int(numerator[k])

  return result
# nCr(2) --- END ---

# Struct -- START --
class struct:
  def __init__(self,a,b,c):
    self.a=a
    self.b=b
    self.c=c
# Struct --- END ---

# Elatostenes's sieve -- START --
def elatostenesSieve(n):
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
# Elatostenes's sieve --- END ---

# primes -- START --
def primes(x):
  if x<2:
    return []

  a=[i for i in range(x)]
  a[1]=0

  for b in a:
    if b>math.sqrt(x):
      break
    if b==0:
      continue
    for c in range(2*b,x,b):
      a[c]=0

  return [b for b in a if b!=0]
# primes --- END ---

# nsPrimes -- START --
def nsPrimes(n):
  ret=[]
  while n%2==0:
    ret.append(2)
    n//=2
  b=3
  while b*b<=n:
    if n%b==0:
      ret.append(b)
      n//=b
    else:
      b+=2
  if n!=1:
    ret.append(n)
  return ret
# nsPrimes --- END ---

# Factoring by trial split -- START --
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
# Factoring by trial split --- END ---

# isPrime -- START --
def isPrime(x):
  if x==1:
    return False
  for i in range(2,int(math.sqrt(x))+1):
    if x%i==0:
      return False
  return True
# isPrime --- END ---

# 10 -> n -- START --
def ten2n(a,n):
  x=a//n
  y=a%n
  if x:
    return ten2n(x,n)+str(y)
  return str(y)
# 10 -> n --- END ---

## usage example of ten2n -- START --
for i in range(pow(2,n)):
  print(ten2n(i,2).zfill(n))
## usage example of ten2n --- END ---

# Union-Find -- START --
class UnionFind():
  def __init__(self,sz):
    self.sz=sz
    self.data=[-1]*sz
    self.amount=[0]*sz

  def unite(self,x,y):
    x=self.find(x)
    y=self.find(y)
    if x==y:
      return False
    self.amount[x]+=self.amount[y]
    self.amount[y]+=self.amount[x]
    if self.data[x]>self.data[y]:
      x,y=y,x
    self.data[x]+=self.data[y]
    self.data[y]=x
    return True

  def find(self,k):
    if self.data[k]<0:
      return k
    self.data[k]=self.find(self.data[k])
    return self.data[k]

  def size(self,k):
    return -self.data[self.find(k)]

  def set_amount(self,k,k_amount):
    self.amount[k]=k_amount

  def get_amount(self,k):
    return self.amount[k]
# Union-Find --- END ---

# Convert from decimal to N -- START --
def Base10ToN(x,n):
  ret=''
  while x>0:
    ret=str(x%n)+ret
    x=int(x/n)
  return ret
# Convert from decimal to N --- END ---

# 進数変換コード例　-- START --
def eight2Ten(n):
  ret=0
  n=str(n)[::-1]
  for i in range(len(n)):
    ret+=int(n[i])*(8**i)
  return ret

def ten2Nine(n):
  ret=''
  while True:
    _n=int(n)
    if _n<9:
      ret+=str(_n)
      return int(ret[::-1])
    ret+=str(n%9)
    n//=9
# 進数変換コード例　--- END ---

# Analog clock -- START --
class AnalogClock():
  def __init__(self,h,m):
    self.h=h
    self.m=m
    self.SHDegree=30*h+m/2
    self.LHDegree=6*m
# Analog clock --- END ---

# Beaufort scale -- START --
l=[0.2,1.5,3.3,5.4,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6]
# A~Z --- END ---

# keyboard -- START --
s='WBWBWWBWBWBW'
t=['Do','Do','Re','Re','Mi','Fa','Fa','So','So','La','La','Si']
# keyboard --- END ---

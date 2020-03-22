# Summarize count of factor within list -- START --
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
# Summarize count of factor within list --- END ---

# nC2 -- START --
def nC2(n):
  return n*(n-1)//2
# nC2 --- END ---

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

# a〜z -- START --
l=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# a〜z --- END ---

# A~Z -- START --
l=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# A~Z --- END ---

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

# nCr -- START --
def nCr(n,r):
  if n<r:
    return 0
  return math.factorial(n)//(math.factorial(n-r)*math.factorial(r))
# nCr --- END ---

# nCr(2) -- START --
def cmb(n, r):
  if n - r < r: r = n - r
  if r == 0: return 1
  if r == 1: return n

  numerator = [n - r + k + 1 for k in range(r)]
  denominator = [k + 1 for k in range(r)]

  for p in range(2,r+1):
    pivot = denominator[p - 1]
    if pivot > 1:
      offset = (n - r) % p
      for k in range(p-1,r,p):
        numerator[k - offset] /= pivot
        denominator[k] /= pivot

  result = 1
  for k in range(r):
    if numerator[k] > 1:
      result *= int(numerator[k])

  return result
# nCr(2) --- END ---

# Struct -- START --
class struct:
  def __init__(self,a,b,c):
    self.a=a
    self.b=b
    self.c=c
# Struct --- END ---

# Node -- START --
class node(parent,left,right):
  def __init__(self,parent,left,right):
    self.parent=parent
    self.left=left
    self.right=right
# Node --- END ---

# Elastotenes's sieve -- START --
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
# Elastotenes's sieve --- END ---

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

# Fibonacci DP -- START --
dp=[0]*110
dp[0]=1
dp[1]=1

def fib(n):
  if dp[n]!=0:
    return dp[n]
  dp[n]=fib(n-1)+fib(n-2)
  return dp[n]
# Fibonacci DP --- END ---

# LCS -- START --
def lcs(x,y):
  m=len(x)
  n=len(y)
  dp=[[0]*(n+1) for _ in range(m+1)]

  for i in range(m):
    for j in range(n):
      if x[i]==y[j]:
        dp[i+1][j+1]=dp[i][j]+1
      else:
        dp[i+1][j+1]=max(dp[i][j+1],dp[i+1][j])

  return dp[m][n]
# LCS --- END ---

# Union-Find -- START --
par=[0]*100010
rank=[0]*100010

## initialize with n elements
def init(n):
  for i in range(n):
    par[i]=i
    rank[i]=0

## find root of tree
def find(x):
  if par[x]==x:
    return x
  else:
    par[x]=find(par[x])
    return par[x]

## unite the set to which x and y belong
def unite(x,y):
  x=find(x)
  y=find(y)
  if x==y:
    return

  if rank[x]<rank[y]:
    par[x]=y
  else:
    par[y]=x
    if rank[x]==rank[y]:
      rank[x]+=1

## whether x and y belong to the same set
def same(x,y):
  return find(x)==find(y)
# Union-Find --- END ---

# Convert from decimal to N -- START --
def Base10ToN(x,n):
  ret=''
  while x>0:
    ret=str(x%n)+ret
    x=int(x/n)
  return ret
# Convert from decimal to N --- END ---

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

# keta DP -- START --
def main():
  s=S()
  l=len(s)

  dp=[[[0]*2 for _ in range(2)]for __ in range(l+1)]
  dp[0][0][0]=1
  
  for i in range(l):
    n=int(s[i])
    for j in range(2):
      for k in range(2):
        if j==0:
          for m in range(n+1):
            dp[i+1][j or m<n][k or m==3]+=dp[i][j][k]
        else:
          for m in range(10):
            dp[i+1][j][k or m==3]+=dp[i][j][k]

  return dp[l][0][1]+dp[l][1][1]
# keta DP --- END ---

# Beaufort scale -- START --
l=[0.2,1.5,3.3,5.4,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6]
# A~Z --- END ---

# keyboard -- START --
s='WBWBWWBWBWBW'
t=['Do','Do','Re','Re','Mi','Fa','Fa','So','So','La','La','Si']
# keyboard --- END ---
import sys
import os
import re

n=6
a=list([1,1,1,2,2,2])


def gcd(a,b):
	a, b = (a, b) if a >=b else (b, a)
	while b:
		a,b = b,a%b
	return a	

def  main(n,a):

	count_dict = dict()
	for item in a:
		if item in count_dict:
			count_dict[item] += 1
		else:
			count_dict[item] = 1
#	print(count_dict)

	a_sorted=sorted(count_dict.values())
	print(a_sorted)
#    sd=zip(a_sorted, a_sorted[1:] + a_sorted[:1])
#    print(sd)
	for i in range(len(a_sorted)):
		for j in range(i,len(a_sorted)):
			m=a_sorted[i]
			n=a_sorted[j]
			if i==0 & j==0:
				gcd_now=gcd(m,n)

			gcd1=gcd(m,n)
			if gcd(gcd1,gcd_now)==1:
				return(0)
			else:
				gcd_now=gcd(gcd1,gcd_now)
	return(sum(a_sorted)/gcd_now)

#    for item in sd:

#    print(a_sorted)
print(main(n,a))
print(gcd(3,3))
#    a, b = (a, b) if a >=b else (b, a)
#    while b:
#        a,b = b,a%b
#    return a

num=int(input("Enter the number: "))

divList=[divNum for divNum in range(1,num//2) if num%divNum==0]
divList.append(num)
print(divList)


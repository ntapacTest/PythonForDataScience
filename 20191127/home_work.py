def getLetters(inString):
    lstAll=set(list(inString))
    letters={'a','e','i','o','u'}
    result=list(lstAll.intersection(letters))
    result.sort()
    return result


inputString=input("Enter string: ")
result=getLetters(inputString)
print(result)

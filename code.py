nav=int(input("enter a list of numbers separated by commas: "
              )
)
max=[]
for n in nav:
    if n>max:
        max=n
    else:
        max=max
print("The maximum number is:", max)
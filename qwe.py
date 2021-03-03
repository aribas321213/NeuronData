a = int(input())
b = int(input())+1
for i in range(a, b):
    ans = []
    for j in range(1, i+1):
        if i % j == 0:
            ans.append(j)
        if len(ans) > 6:
            break
    if len(ans) == 6:
        print(*ans)

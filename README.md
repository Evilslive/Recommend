## Recommend
推薦系統 / 記錄心得 / 提醒自己


### Collaborative Filtering
最常提到的「協同演算法」，有以下幾種基底方法
+ Memory-based &emsp; 
+ Model-based &emsp; 
+ User-based &emsp; 尋找與相同愛好的使用者
+ Item-based &emsp; 購物的先後順序 (買了薯條、再買番茄醬)
+ User&Item-based &emsp; 同時考慮人與物品, 如KMeans


### Pandas 操作

> *.read_csv* &emsp; 讀取gzip壓縮檔

```python
df = pd.read_csv(url, 
                 compression='gzip',
                 header=0, # 標頭列
                 skip_blank_lines=True, 
                 skiprows=[], # 跳過多少行
                 nrows=100) # 讀取多少行
```

> *.crosstab* &emsp; 產生資料交叉表

```python
table = pd.crosstab(df.chid, # 縱列 index
                    df.shop_tag, # 橫欄 column
                    aggfunc=np.mean, # 計算公式
                    values=df.txn_amt) # cell內的值
```

> *.drop* &emsp; Drop specified labels from rows or columns.

```python
table = table.drop(columns=100) # 排除欄
```

> *.DataFrame* &emsp; Two-dimensional, size-mutable, potentially heterogeneous tabular data.

```python
base = pd.DataFrame() # 產生 空 資料框架
base['chid'] = table.index # 將table的index建為欄'chid'
base['tags'] = shop # 將 shop:list 建為欄'tags'
```
&nbsp;
### *Longest common Subsequence with DP*
最長共同序列問題, 除了取出最長共有子序外, 也要提供子序最後結束的位置(作為事件切分點)

```python
def lcs(m:list, n:list):
    ''' 動態規劃 '''
    dp = [[(0,"")]*(len(n)+1) for _ in range(len(m)+1)] # 產生保存矩陣
    for s,t in enumerate(m):
        for v,w in enumerate(n):
            if dp[s][v][0] >= max(dp[s+1][v][0], dp[s][v+1][0]):
                dp[s+1][v+1] = (dp[s][v][0], "d") # 取對角線值 diagonal
            elif dp[s+1][v][0] > dp[s][v+1][0]:
                dp[s+1][v+1] = (dp[s+1][v][0], "l") # 取左側值 left
            elif dp[s][v+1][0] > dp[s+1][v][0]:
                dp[s+1][v+1] = (dp[s][v+1][0], "u") # 取上方值 up
            else: 
                # 如果上左都可以走的話, ex: (1,3,6,5) vs (1,6,3,5)
                dp[s+1][v+1] = (dp[s][v+1][0], "x") # 岔路標記, 後續upsteam疊代 (1)
            # 找到共同
            if t == w: 
                dp[s+1][v+1] = (dp[s+1][v+1][0]+1, dp[s+1][v+1][1])
    
    result = upstream(dp) # 逆流而上, 得出最長共用子序列
    if result == [[]]: return [[]], [[]] # 沒找到
    return [tuple(m[j[0]-1] for j in i) for i in result], \
           [(max([j[0] for j in i]), max([j[1] for j in i])) for i in result] # 結束點cut point
           
def upstream(matrix:list, m:int=None, n:int=None, cache:list=[[]])->list:
    ''' cache: [[(m1,n1), (m2, m2)]]
        希望可以岔路遍歷, 雖然都是最長共子序, 但特徵可能有不同的意義
        ex: m=(1,3,6,5) n=(1,6,3,5), 有兩個共同字序 (1,3,5), (1,6,5)
        
        note: 這裡未考慮重複字串的問題
    '''
    if m == None: m = len(matrix)-1
    if n == None: n = len(matrix[0])-1
    dcopy = []
    dcopy.extend(cache)
    if matrix[m][n][1] == "":
        return cache # output
    elif matrix[m][n][1] == "u":
        if matrix[m][n][0] > matrix[m-1][n][0]:
            dcopy[-1] = [(m,n)] + dcopy[-1]
        return upsteam(matrix, m-1, n, dcopy)
    elif matrix[m][n][1] == "l":
        if matrix[m][n][0] > matrix[m][n-1][0]:
            dcopy[-1] = [(m,n)] + dcopy[-1]
        return upsteam(matrix, m, n-1, dcopy)
    elif matrix[m][n][1] == "x": # 岔路, 先走一條, 再走另一條 (1)
        if matrix[m][n][0] > matrix[m-1][n][0]:
            dcopy[-1] = [(m,n)] + dcopy[-1]
        one_of = upsteam(matrix, m-1, n, dcopy) # 先走完
        if matrix[m][n][0] > matrix[m][n-1][0]:
            cache[-1] = [(m,n)] + cache[-1]
        return upsteam(matrix, m, n-1, one_of+cache)
    else:
        # 除非*數字重複*, 不然只會在斜線出現目標
        if matrix[m][n][0] > matrix[m-1][n-1][0]:
            dcopy[-1] = [(m,n)] + dcopy[-1]
        return upsteam(matrix, m-1, n-1, dcopy)
```



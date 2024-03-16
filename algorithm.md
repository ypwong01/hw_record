[TOC]

## 动态规划

动态规划是一种用于求解最优化问题的数学方法。它的基本思想是将一个复杂的问题分解成一系列简单的子问题（称为状态），并且记录下各个子问题的解，以便于下次需要解决同一个子问题时，可以直接使用已经记录的解，避免了重复计算。

动态规划的步骤通常包括：

1. **定义状态**：将原问题分解成一系列子问题，每个子问题对应一个状态。
2. **状态转移方程**：找出状态之间的关系，即如何从已知的状态得到未知的状态。
3. **初始化**：确定初始状态的值。
4. **计算状态**：根据状态转移方程和初始状态，按照一定的顺序计算出所有状态的值。
5. **返回结果**：根据计算出的状态值，得到原问题的解。

动态规划在很多问题中都有应用，例如背包问题、最长公共子序列、石子合并问题等。动态规划适用于具有“最优子结构”和“无后效性”的问题，即子问题的最优解能够用于求解原问题的最优解，且子问题的解不会影响到其他子问题的解。

### 背包问题

#### 0-1背包

P5011

二维

```cpp
 for(int i=1;i<=n;i++){
        for(int j=i;j<=v;j++){
           if(j>=weights[i])  dp[i][j] = max(dp[i-1][j],dp[i-1][j-weights[i]]+value[i]);
           else dp[i][j] = dp[i-1][j];
        }
    }
```

一维

```
for(int i=1;i<=n;i++){
    for(int j=v;j>=weights[i];j--){
        dp[j] = max(dp[j],dp[j-weights[i]]+value[i]);
    }
}
eg dp[3] = max(dp[3],dp[2]+1)
正向遍历的话，更新到dp[3]的时候，dp[2]已经被更新过了，破坏了dp[3]的更新
    
优化：
由于dp[i][j]仅依赖于上一行且列数小于等于j的状态，可优化为一维数组
空间复杂度优化从O(NV)到O(V)
注意背包容量需要从后向前遍历
从前向后遍历，一个物品可能会被使用多次
物品需要放在外层循环，才可以决定是否使用，在哪里使用
```

#### 完全背包

P5013

0-1背包是完全背包的一种特殊情况

二维

```cpp
dp[i][j]表示前i种物品放入容量为j的背包的最大价值
    for(int i=1;i<=n;i++){
        for(int j=1;j<=v;j++){
            if(j>=weights[i]){
                for(int k=0;k*weights[i]<=j;k++){
                    dp[i][j] = max(dp[i-1][j],dp[i-1][j-k*weights[i]]+k*value[i]);
                }
            }else{
                dp[i][j] = dp[i-1][j];
            }
        }
    }
```

一维

```cpp
for(int i=1;i<=n;i++){
        for(int j=weights[i];j<=v;j++){
            dp[j] = max(dp[j],dp[j-weights[i]]+value[i]);
        }
    }
```

#### 多重背包

P5012

二维

```cpp
tips:k * weights[i]<=j  且 k可以取到0
    
    for(int i=1;i<=n;i++){
        for(int j=1;j<=v;j++){
            if(j>=weights[i]){
                for(int k=0;k<=q[i] && k*weights[i]<=j;k++){
                    dp[i][j] = max(dp[i-1][j],dp[i-1][j-k*weights[i]]+k*value[i]);
                }
            }else{
                dp[i][j] = dp[i-1][j];
            }
        }
    }    
    
```

k=1开始

```cpp
k从1开始，需要在前边补充k=0，也就是不取第i种物品的情况
for(int i=1;i<=n;i++){
        for(int j=1;j<=v;j++){
            dp[i][j] = dp[i-1][j];//k=0的情况
            for(int k=1;k<=q[i] && k*weights[i]<=j;k++){
                //这里一直不断max就是在更新结果，判断取多少件能够获取最大价值
                //不取第i个物品：k=0
                dp[i][j] = max(dp[i][j],dp[i-1][j-k * weights[i]]+k * value[i]);
            }
        }
    }
```

一维

```cpp
  for(int i=1;i<=n;i++){
        for(int j=V;j>=weights[i];j--){
            //下面这一层循环的核心是取几个物品
            for(int k=0;k<=q[i] && k*weights[i]<=j;k++){
                dp[j] = max(dp[j],dp[j-k*weights[i]]+k*value[i]);
            }
        }
    } 
```

#### 补充：分组背包

分组背包问题是背包问题的一种变形，每种物品不再只有一个，而是分成了若干组，每组里面的物品互斥，只能选一个。

```cpp
可以将分组背包看作多重背包，把每组看成一个物品，物品可以选择0到si个，通过从后向前的遍历顺序来确保，对组的决策只有一种：要么选这个组，要么不选；然后再通过枚举组内的情况，来对组内进行决策：要么选0个，选1个.....；
for (int i = 1; i <= n; i ++ )
    for (int j = v; j >= 0; j -- )
        for (int k = 0; k <=s[i]; k ++ )
            if (v[i][k] <= j)
                f[j] = max(f[j], f[j - v[i][k]] + w[i][k]);
```

#### 补充：部分背包

部分背包 VS 0-1背包：部分背包可以拿走部分，涉及按照单位重量价值对物品排序，使用贪心算法可解决。

### 线性DP

P5003最大连续子数组和

```cpp
思路：
1.定义dp[i]为以nums[i]结尾的最大连续子数组和
2.初始化  将dp每个元素初始化为nums[i]，状态转移依赖于i-1，所以事实上初始化最初的就可以
for(int i=0;i<n;i++) dp[i] = nums[i];
3.状态转移 
dp[i] = max(dp[i-1]+nums[i],nums[i])
4.结果是dp数组中所有元素最大值，ans = * max_element(dp,dp+n);

时间复杂度：
O(n)
    
PS：分治法时间复杂度O(nlogn) 
```

P5004最长递增子序列

```cpp
思路
1.定义dp[i]为以nums[i]结尾的最长递增子序列的长度
2.初始化:全部初始化为1
3.状态转移
for(int i=1;i<=n;i++){
	for(int j=0;j<i;j++){
		if(nums[i]>nums[j]) dp[i] = max(dp[i],dp[j]+1);
		//这里max中有dp[i]是因为随着遍历j,dp[i]会不断更新
	}
}
时间复杂度:O(n^2)
```

P5005打家劫舍

```cpp
for(int i=1;i<=n;i++) cin >> nums[i];
dp[1] = nums[1];
dp[2] = max(nums[1],nums[2]);
for(int i=3;i<=n;i++) dp[i]= max(dp[i-1],dp[i-2]+nums[i]);
```

### 二维DP

P5006最长公共子序列

```cpp
开(n+1)*(m+1)的dp  呈现一个格子的形状  dp[i][j]一般依赖于三个方向
定义：dp[i][j]表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符的LCS长度。
初始化：边界初始化为0即可
状态转移：字符相同时，LCS长度为
for(int i=1;i<=n;i++){
	for(int j=1;j<=m;j++){
		if(s1[i-1]==s2[j-1]){
			dp[i][j] = dp[i-1][j-1] + 1;
		}else{
			dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
		}
	}
}
```

P5007字符串的编辑距离

```cpp
与LCS问题同类
dp[i][j] 表示字符串 A 的前 i 个字符转换为字符串 B 的前 j 个字符所需的最小操作数。
初始化：第一行全部初始化为j,第一列全部初始化为i
for(int i=0;i<=n;i++){
    dp[i][0] = i;
}
for(int j=0;j<=m;j++){
    dp[0][j] = j;
}
for(int i=1;i<=n;i++){
	for(int j=1;j<=m;j++){
		if(s1[i-1]==s2[j-1]){
			dp[i][j] = dp[i-1][j-1]
		}else{
			dp[i][j] = min(min(dp[i-1][j],dp))
		}
	}
}
```

P5001数字三角形

```cpp
自底部向上遍历
    //初始化最底部
    for(int j=1;j<=n;j++){
        dp[n][j] = tri[n][j];
    }
    //状态转移
    for(int i=n-1;i>=1;i--){
        for(int j=1;j<=i;j++){
            dp[i][j] = max(dp[i+1][j+1],dp[i+1][j]) + tri[i][j];
        }
    }
```

### 区间DP

区间DP用于处理与序列相关的问题。

P5015石子合并

```cpp
思路：
1.定义dp[i][j]为合并从i到j堆石子的总代价
2.由于要求最小值，初始化为很大的值
3.由于最后一定是合并两堆，所以拆分为i到k和k+1到j，再加上这两堆合并的代价
PS:  VS 果子合并
本题不可以使用优先队列 因为需要是相邻的石子堆，果子合并不作要求
for(int i=1;i<=n;i++){
    cin >> nums[i];
    sum[i] += sum[i-1] + nums[i];
}
//dp[i][j]代表合并从i到j堆的重量
for(int len=2;len<=n;len++){
    for(int i=1;i<=n-len+1;i++){
        int j=i+len-1;
        dp[i][j] = 2e9;
        for(int k=i;k<j;k++){
            dp[i][j] = min(dp[i][k]+dp[k+1][j]+sum[j]-sum[i-1],dp[i][j]);
        }
    }
}
```

### 序列决策

P5014股票交易

```cpp
1.初始化
dp[0][0] = 0;
dp[0][1] = -prices[0];
2.状态转移
分为两种状态：第i天持有股票和不持有股票  
不持有股票：最大利润可以是前一天就不持有  or  前一天就持有，今天卖出
dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i]);
持有股票：最大利润可以是前一天就持有  or  前一天不持有，今天买入
dp[i][1] = max(dp[i-1][1],dp[i-1][0]-prices[i]);
每天更新这两个状态的最大利润
3.
第n天不持有股票的最大利润即为所求，即dp[n-1][0]
```

## 分治

分治策略是一种解决问题的方法，它将一个复杂的问题分解成两个或更多的相同或相似的子问题，直到最后子问题可以简单的直接求解，原问题的解即子问题的解的合并。

分治策略通常可以通过递归实现，其基本步骤包括：

1. **分解**：将原问题分解成一系列子问题；
2. **解决**：如果子问题足够小，那么直接解决。如果子问题还是复杂的，那么再次应用分治策略；
3. **合并**：将子问题的结果合并成原问题的解。

分治策略适用于子问题相互独立，且与原问题形式相同的情况。如果子问题之间有重叠部分，那么分治策略可能会做很多不必要的工作，因为它会重复求解那些重叠的子问题。在这种情况下，通常使用动态规划或记忆化搜索会更有效。

归并排序、快速排序都是分治策略的典型应用：

P6001归并排序+P6002逆序对的数量

```cpp
归并排序的步骤：
分解：将待排序的数组从中间分成两半，形成两个子数组。如果数组只有一个元素，那么它已经是有序的，无需再分。
解决：使用归并排序递归地排序两个子数组。
合并：将两个已排序的子数组合并成一个有序的数组。
void mergesort(int q[],int l,int r){
    if(l>=r) return;
    int mid = l+r>>1;
    mergesort(q,l,mid),mergesort(q,mid+1,r);
    //merge
    int i=l,j=mid+1,k=0;

    while(i<=mid && j<=r){
        if(q[i]<q[j]) tmp[k++] = q[i++];
        else tmp[k++] = q[j++];
    }
    while(i<=mid) tmp[k++] = q[i++];
    while(j<=r) tmp[k++] = q[j++];

    //copy
    for(int i=l,k=0;i<=r;i++,k++) q[i] = tmp[k];
}

求逆序对的数量：
形成逆序对时，两个元素i和j可能位于三个部分：左半边，右半边，跨越中点
跨越中点部分在合并merge过程中计算出来
ll mergesort(int q[],int l,int r){
    if(l>=r) return 0;
    int mid = l+r>>1;
    ll ans = 0;
    //mergesort左右两边
    ans += mergesort(q,l,mid);
    ans += mergesort(q,mid+1,r);
    //merge
    int i=l,j=mid+1,k=0;
    while(i<=mid && j<=r){
        if(q[i]<=q[j]) tmp[k++] = q[i++];
        else{
            tmp[k++] = q[j++];
            ans += (mid-i+1);
        }
    }
    while(i<=mid) tmp[k++] = q[i++];
    while(j<=r) tmp[k++] = q[j++];
    //copy
    for(int i=l,k=0;i<=r;i++,k++){
        q[i] = tmp[k];
    }
    return ans;
}
```

快速排序+P6003快速选择第k小的数字

```cpp
快速排序：
分解：选择一个元素作为"基准"，重新排列数组，使得所有比基准小的元素都在基准之前，所有比基准大的元素都在基准之后。在这个分区结束之后，基准就处于数组的最终位置。
解决：通过递归调用快速排序，对基准前后的两个子数组进行排序。
合并：因为子数组都是原址排序的，所以不需要合并操作：一旦快速排序返回，数组就已经被排序了。
void quicksort(int q[],int l,int r,int k){
    if(l>=r) return q[k-1];//这个就是最终的结果了
    int i = l-1,j=r+1;
    int x = q[l];
    while(i<j){
        do i++; while(q[i]<x);
        do j--; while(q[j]>x);
        if(i<j) swap(q[i],q[j]);
    }
    quicksort(q,l,j,k);
    quicksort(q,j+1,r,k);
}

快速选择：
注意index问题：第k小的数字index是k-1
int quicksort(int q[],int l,int r,int k){
    if(l>=r) return q[k-1];//这个就是最终的结果了
    int i = l-1,j=r+1;
    int x = q[l];
    while(i<j){
        do i++; while(q[i]<x);
        do j--; while(q[j]>x);
        if(i<j) swap(q[i],q[j]);
    }
    //如果k小于等于j,说明第k小的数字在左半边
    if(k<=j) return quicksort(q,l,j,k);
    else return quicksort(q,j+1,r,k);
}
```

P6005最大连续子数组和

```cpp
分别求左半边的最大连续子数组和，右半边的最大连续子数组和，以及横跨左右两边的最大连续子数组和，取最大值
int fun(int l,int r){
    //求横跨左右两边的
    int mid = l + ((r-l)>>1);
    int suml = 0,ansl = nums[mid];
    for(int i=mid;i>=l;i--){
        suml+=nums[i];
        ansl = max(ansl,suml);
    }
    int sumr = 0,ansr = nums[mid+1];
    for(int i=mid+1;i<=r;i++){
        sumr += nums[i];
        ansr = max(ansr,sumr);
    }
    return ansl + ansr;
}

int max_sum(int l,int r){
    if(l==r) return nums[l];
    int lr = fun(l,r);
    int mid = l + ((r-l)>>1);
    int left_sum = max_sum(l,mid);
    int right_sum = max_sum(mid+1,r);
    int all_max = max(max(left_sum,right_sum),lr);
    return all_max;
}
```

## 二分

同样是分治策略的运用。二分查找的特点：

1. **单调性**：问题的解在一个范围内单调递增或递减。例如，如果你要找的是一个数的平方根，那么随着数的增加，它的平方根也是增加的。

2. **二段性（二分性）**：问题的答案可以按照“是或否”来二分。如果可以定义一个函数 check(int x)使得在某个点之前 check(int x) 都返回“是”，而之后都返回“否”，或者相反，那么这个问题就有二分性。这样你就可以通过不断地缩小“是”和“否”的边界来找到这个分界点。

3. **有界性**：二分搜索需要在一个确定的、有限的范围内进行，需要能够定义出搜索的上界和下界。

4. **可通过索引访问**：需要随机访问元素，问题的解空间应该支持快速访问，即能在 O(1) 时间内根据索引取得值，这样才能在二分搜索中有效地选取中点并比较，如果是链表这种只能进行顺序访问的数据结构，效率可能会很低。

5. **可计算的中值和判断条件**：需要能够计算出解空间的中值，以及在该中值上能应用判断条件来决定应该向左还是向右继续搜索。

   PS：随机访问的数据结构：是指可以在常数时间内访问其任意位置的元素的数据结构。也就是说，无论元素在数据结构中的位置如何，访问该元素的时间复杂度都是O(1)，比如数组、哈希表。

### 整数二分

P6011查找翻转子序列最小值

```cpp
int l=0,r=n-1;//n是数组长度
while(l<r){
	 int mid = l+r>>1;
     if(q[mid]>q[r]) l = mid + 1;//说明最小值在右半边,更新l
     else r = mid;
}
cout << q[r] << endl;
```

P6012有序数组中查找两个数字

```cpp
分别查找左右边界：
左边界的定义——所有大于等于target的值都在leftborder（包含）右边，小于target的值都在leftborder左边
右边界的定义——所有小于等于target的值都在rightborder（包含）左边，大于leftborder的值都在rightborder右边
int getLeft(int target){
    int l=0,r=n-1;
    while(l<r){
        int mid = (l+r)>>1;
        if(q[mid]>=target){
            r = mid;
        }else{
            l= mid +1;
        }
    }
	return (nums[l]==target?l:-1)
}
int getRight(int target){
   	int l=0,r=n-1;
    while(l<r){
        int mid = (l+r+1)>>1;
        if(q[mid]<=target){
            l=mid;
        }else{
            r=mid-1;
        }
    }
	return (nums[r]==target?r:-1)
}
```

一些问题在给定条件下寻求左右边界，需要基于给定条件确定check函数，比如砍树和数列分段问题。

P6014砍树

```cpp
思路：
在满足要求的段数m的情况下，最大化单位木材的长度——需要check 段数是否大于等于m

1.确定单位木材长度上下界 int l=0,int r= *max_element(height,height+n);
2.二分+check
核心代码
bool check(int x){
//参数x是单位木材的长度
	int cnt = 0;
	for(int i=0;i<n;i++){
		cnt += height[i]/x;
	}
	return cnt>=m；
}
while(l<r){
	int mid = 
	if(check(mid)){
		//说明还可以向右边找-增加长度
      	l = mid;
	}else{
        r = mid-1;
    }
}
3.输出 cout << l << endl;
```

P6015数列分段

```cpp
思路：给定段落数（下限）要求的情况下，最小化每一段的和
1.确定上下界 int l=*max_element(nums,nums+n);int r = 数列所有元素之和
2.二分+check
bool check(int x){
	int sum = 0;
	for(int i=0;i<n;i++){
		if(sum+nums[i]>x){
			cnt++;
			sum = nums[i];
		}else{
			sum += nums[i];
		}
	}
    return cnt<=m;
}
while(l<r){
    int mid = (l+r+1)>>1;
    if(check(mid)){
        r=mid;
    }else{
        l=mid+1;
    }
}
3.输出 cout << r << endl;
```

### 浮点二分

P6013数的三次方根

```cpp
double n;
double l=0,r=n-1;
while(r-l>1e-8){
	double mid = (l+r)/2; //这里一定是double
	if(mid * mid *mid <=x){
		l = mid;
	}else{
		r = mid;
	}
}
printf("%lf\n",l);
时间复杂度:O(n) n-查找范围
```

## 贪心

贪心算法是一种在每一步选择中都采取在当前状态下最优的选择，从而希望导致结果是最好或最优的算法。贪心算法的主要特点是：它总是做出在当前看来是最好的选择，也就是说它不从整体最优上加以考虑，它所做出的是在某种意义上的局部最优解。贪心算法只对问题的局部做出选择，但并不保证这些选择能够达到全局最优。

典型例题如下：

P7001采购礼物

```cpp
将
int cnt = 0;
for(int i=0;i<n;i++) cin >> prices[i];
sort(prices,prices+n);
for(int i=0;i<n;i++){
    if(m>=prices[i]){
        cnt++;
        m-=prices[i];
    }
}
```

P7002硬币找零

```cpp
 //for循环里边不需要判断  
const int prices[5] = {50,20,10,5,1};
int m = 100-n;
int ans = 0;
for(int i=0;i<5;i++){
    ans += m/prices[i];
    m = m%prices[i];
}
cout << ans;
```

P7011股票交易

```cpp
for(int i=2;i<=n;i++){
	if(prices[i]>prices[i-1]){
		ans += prices[i]-prices[i-1];
	}
}
```

P7003部分背包

```cpp
for(int i=0;i<n;i++){
    cin >> a[i].m >> a[i].v;
    a[i].p = 1.0 * a[i].v/a[i].m;//注意这里的1.0
}
sort(a,a+n,cmp);//定义cmp——降序排列  按照结构体的p成员变量
double ans = 0;
for(int i=0;i<n;i++){
    if(t>=a[i].m){
        ans+=a[i].v;
        t-=a[i].m;
    }else{
        ans+=t*a[i].p;
        break;
    }
}
```

P7007数列分段

```cpp
二分中的check函数部分:即如果目前段的元素和加上新的元素nums[i]>x 那么就再加一段，并将sum置为nums[i]
否则，将这一个新的元素加入到当前这段数列中
int cnt = 1;
int sum = 0;
for(int i=0;i<n;i++){
    if(sum+nums[i]>m){
        cnt++;
        sum = nums[i];
    }else{
        sum += nums[i];
    }
}
```

P7005区间选点

```cpp
选择区间右端点作为选点可以“覆盖”最多的区间，减少所需的总点数。
因而根据右端点排序，遍历所有区间，每次选择当前区间的右端点作为选点，然后跳过所有包含这个点的区间。
sort(A,A+n,cmp);
int be = -2e9;
int cnt = 0;
for(int i=0;i<n;i++){
    if(A[i].b>be){
        cnt++;
        be = A[i].e;
    }
}
```

P7006最优活动安排

```
除端点部分的处理外，和区间选点属于同一类型。
```

P7007区间覆盖

```cpp
给定需要覆盖的区间起始点start和终点tail
尽量找到最少的区间：
所有区间按照左端点排序
遍历所有能够覆盖start的区间,找到右端点最大的那个maxEnd，更新maxEnd
/* 
1.根据起始点排序
2.遍历所有区间，找能够覆盖start的区间，也就是begin<start的区间  这一部分是通过while循环实现的
1)if 不能够找到 也就是maxEnd < start，就success=false,break
2)else 能够找到，更新maxEnd 为新的start,cnt++
  如果最后maxEnd >= tail 说明成功了
*/
核心代码：
for(int i=0;i<n;i++){
    int j=i;
    int maxEnd = -2e9;//放在这里是因为要对每一个更新后的start去找maxEnd
    while(j<n && A[j].b<=start){
        maxEnd = max(maxEnd,A[j].e);
        j++;
    }
    if(maxEnd<start){
        success = false;
        break;
    }else{
        //更新start,区间数+1
        start = maxEnd;
        cnt++;
        if(maxEnd>=tail){//或者start>=tail
            success = true;
            break;
        }
    }
```

P7012排队取水——排序不等式

```cpp
sort(nums,nums+n);
for(int i=0;i<n;i++){
    ans += (n-1-i) * nums[i];
}
```

P7013仓库选址——绝对值不等式

```cpp
sort(nums,nums+n);
for(int i=0;i<n;i++) ans += abs(nums[i]-nums[(n-1)/2]);
```

P7014均分图书——平均值

```cpp
int avg = sum/n;
int cnt = 0;
for(int i=0;i<n;i++){
    if(q[i]!=avg){
        q[i+1] += q[i]-avg;
        q[i] = avg;
        cnt++;
    }
}
```

P7015果子合并 —— Huffman树

```cpp
while(pq.size()>1){
    int a = pq.top();pq.pop();
    int b = pq.top();pq.pop();
    ans += a+b;
    pq.push(ans);
}
```

## 回溯

P3003枚举子集1

P8001枚举子集2

```cpp
是一类问题，一种是枚举数字——用startIndex表示，一种是枚举对应索引的数组元素

枚举子集1：
vector<int> path;//用于收集结果
void print(){
    for(int i=0;i<path.size();i++){
        cout << path[i] << " ";
    }
    cout << endl;
}
void dfs(int startIndex){
    if(startIndex>n){
        print();
        return;
    }
    path.push_back(startIndex);
    dfs(startIndex+1);
    path.pop_back();
    dfs(startIndex+1);
}

枚举子集2：
void print(){
    for(int i=0;i<path.size();i++){
        cout << path[i] << " ";
    }
    cout << endl;
}
void dfs(int startIndex){
    if(startIndex>n){
        print();
        return;
    }
    path.push_back(nums[startIndex]);
    dfs(startIndex+1);
    path.pop_back();
    dfs(startIndex+1);
}
```

P8002全排列

```cpp
由于1，2 与 2，1也算是一种排列，所以不需要使用startIndex改变遍历起始点
但是元素不可以重复，需要使用used数组记录
void dfs(bool used[],int nums[]){
    if(path.size()==n){
        print();
        return;
    }
    for(int i=0;i<n;i++){
        if(!used[i]){
            path.push_back(nums[i]);
            used[i] = true;
            dfs(used,nums);
            used[i] = false;
            path.pop_back();
        }
    }
}
```

P3004整数划分——组合问题

```cpp
元素可以重复，终止条件有两个
void dfs(int startIndex,int sum){
    if(sum == n && path.size()==k){
        for(int i=0;i<k;i++){
            cout << path[i] << " ";
        }
        cout << endl;
        return;
    }
    for(int i=startIndex;i+sum<=n;i++){
        path.push_back(i);
        sum += i;
        dfs(i,sum);
        sum -= i;
        path.pop_back();
    }
}
```

P8003N皇后——棋盘问题

```cpp
void dfs(int row,int n,int k){
    if(cnt==k) return;
    if(row==n){
        for(int i=0;i<n;i++) cout << loc[i] + 1 << " ";
        cout << endl;
        cnt++;
        return;
    }
    int x = row;
    for(int y=0;y<n;y++){
        if(!flag1[y] && !flag2[x+y] && !flag3[y-x+n]){
            loc[x] = y;
            flag1[y] = flag2[x+y] = flag3[y-x+n] = true;
            dfs(x+1,n,k);
            flag1[y] = flag2[x+y] = flag3[y-x+n] = false;
        }
    }
}
```

P8004 0-1背包

```cpp

int max_value;
void dfs(int step,int sum_w,int sum_v){
    if(step>n){
        max_value = max(max_value,sum_v);
        return;
    }
    //不放入
    dfs(step+1,sum_w,sum_v);
    //放,需要判断是否合法
    if(sum_w + weights[step]<=v){
        dfs(step+1,sum_w+weights[step],sum_v+value[step]);
    }
}
```

## 分支限界

分支界限法的主要步骤包括：

1. **分支**：将问题的解空间分解为若干个子空间，这通常通过对一个或多个变量进行赋值来实现。
2. **界限**：对每个子空间进行评估，计算一个界限值，这个界限值通常是问题的目标函数的一个上界或下界。如果这个界限值比已知的最优解还要差，那么这个子空间就可以被排除，不需要进一步搜索。

分支界限法的主要特点是：它不仅可以找到问题的一个最优解，而且还可以找到问题的所有最优解。此外，分支界限法还可以用于求解满足一定约束条件的可行解。

典型例题如下：

P8013任务分配

```cpp
解决：
1.可以使用dfs或者dp暴力枚举,但是时间复杂度比较高，dfs具体做法如下：
//g[N][N]为时间矩阵
void dfs(int step,int sum){
    if(sum>=ans) return;
    if(step==n){
        ans = sum;
        return;
    }
    for(int i=0;i<n;i++){
        if(!st[i]){
            st[i] = true;
            dfs(step+1,sum+g[step][i]);
            st[i] = false;
        }
    }
}

2.分支限界法：
void bfs(){
    // 定义一个优先队列 
    priority_queue<Node> p_q;
    Node cur,next;
    cur.i = 0; //根节点
    cur.cost =0;
    cur.used.resize(n+10);
    cur.ve.resize(n+10);
    bound(cur);
    p_q.push(cur); //根节点入队 

    while(!p_q.empty()){ // 队列不为空 
        cur = p_q.top();
        p_q.pop();
        for(int j = 1; j <= n; j++){ //枚举任务 
            if(cur.used[j] == 1) continue; //任务被分配过 
            next.i = cur.i + 1;
            next.used = cur.used;
            next.ve = cur.ve;
            next.used[j] = 1;
            next.ve[next.i] = j; //第i个人，被分配了第j个任务 
            next.cost = cur.cost + a[next.i][j];
            bound(next);//计算下届 
            if(next.lb < ans){//剪枝 
                if(next.i == n){
                    //更新最优解 
                    if(next.cost < ans){
                        ans = next.cost; 
                    }   
                }else{
                    //入队 
                    p_q.push(next); 
                }
            }
        }
    }   
}
```

P8012 0-1背包

```cpp

采用优先队列方式，按照物品的单位价值从大到小进行优先级排序，使用大根堆结构存储物品数据。
构造上界函数maxbound( )计算当前结点下的价值上界，如果当前结点下的价值上界比当前的最优值大，则将当前结点加入堆中，否则剪去该节点下的所有路径(即剪去子集树的枝)，直到堆中所有结点均被弹出。

1.建立状态节点节点需要记录下当前的状态，回溯法并不需要使用节点记录状态
2.明确所有分支  比如：放入背包，不放入背包
3.寻找界限条件:为每一个分支确定界限，不放入——不需要，装入——需要：在物品放进背包之前，判断放入背包是否会超重

上界估算优化：
价值上限=节点现有价值+背包剩余容量*剩余物品的最大单位重量
1）计算已经选择的节点  eg A
2）贪心算法计算未知道路  eg A+B ，A+B+部分C  比较获得最大值，就是A结点的上限

以活结点价值上限——也就是ub为优先准则

所有步骤：
1.创建Item类 or  结构体 ， 定义一个比较函数cmp 根据单位重量的价值进行比较
2.创建状态结点，运算符承载，定义上界高的元素排列在靠前的位置，先出队——记录元素：第i个物品，当前总体积，当前总价值，上界————需要定义上界计算函数：使用贪心算法计算最大价值ub————参考贪心的部分背包问题
3.定义bfs函数
4.输出bestp

代码如下：
void bfs(){
    //1、初始化队列queue ,将第0个物品放入队列 
    Node cur;
    priority_queue<Node> p_q;

    cur.i = 0; //第0个物品 
    cur.cp = 0;
    cur.cv = 0;
    cur.ub  = bound(cur.i+1, cur.cv, cur.cp);//计算上界值 
    p_q.push(cur); //插入优先队列 

    //2.循环遍历队列
    while(!p_q.empty()){ //队列不空 
        // 3.取出队头,存入cur
        cur = p_q.top();
        p_q.pop();

        int n_i = cur.i+1;
        if(n_i > n) continue;

        // 4. 利用产生式规则，拓展cur的关联节点入队
        // 选择下一个物品，并入队 

        if(cur.cv + items[n_i].volume <= V){ // 左剪枝 
            Node t;
            t.i = n_i;
            t.cp = cur.cp + items[n_i].price;
            t.cv = cur.cv + items[n_i].volume;
            t.ub  = bound(cur.i+1, cur.cv, cur.cp);//计算上界值 
            bestp = max(bestp,t.cp);
            p_q.push(t); //插入优先队列 
        } 
        // 不选择下一个物品，并入队 
        // 计算上界
        Node t2;
        t2.i = n_i;
        t2.cp = cur.cp;
        t2.cv = cur.cv;
        t2.ub  = bound(cur.i+1, cur.cv, cur.cp);//计算上界值 

        if (t2.ub > bestp) {
            p_q.push(t2); //插入优先队列 
        }
    } 
} 
```

## 其他

### 双指针

#### 左右指针

P3022有序数组中两数之和

```cpp
//left，right指针不可以同+同-
while(l<r){
    if(nums[l]+nums[r]==target){
        cout << l << " " << r << endl;
        return 0;
    }else if(nums[l]+nums[r]>target){
        r--;
    }else{
        l++;
    }
}
PS：本题也可以使用Hash Map，无序数组中使用Hash Map
```

P3021判断回文字符串

```cpp
//这里的left right 指针必须同+同-
while(l<r){
    if(s[l]!=s[r]){
        cout << "No" << endl;
        return 0;
    }
    l++;
    r--;
}
```

#### 快慢指针

P3023移动数组中的零

```cpp
如果fast指向的是非零的数字，那么快慢指针指向的元素交换，fast一直右移，交换后slow向右边移动，最后结果中非零的都被交换到前边
while(fast<n){
    if(nums[fast]!=0){
        swap(nums[slow],nums[fast]);
        slow++;
    }
    fast++;
}
```

P3024最长连续不重复子序列

```cpp
核心：给定fast指针指向的元素，能向左边最远走到哪里，保证连续子序列不重复——这个长度即为所求
可以借助hash map记录出现的次数
int slow=0,fast=0;
int ans = 0;
unordered_map<int,int> cnt;
while(fast<n){
    cnt[nums[fast]]++;
    while(cnt[nums[fast]]>1){
        cnt[nums[slow]]--;
        slow++;
    }
    ans = max(ans,fast-slow+1);
    fast++;
}
```

### STL与数据结构

#### 队列

特点是FIFO，先进先出，从队头删除元素，队列尾部插入元素。典型应用例题如下：

P2104约瑟夫环

```cpp
queue<int> q;
for(int i=1;i<=n;i++){
    q.push(i);
}
while(!q.empty()){
    for(int i=0;i<m-1;i++){
        q.push(q.front());
        q.pop();
    }
    cout << q.front() << " ";
    q.pop();
}
```

P2004密码翻译

```cpp
vector<int> res;//存储结果
queue<int> q;
cin >> n;
for(int i=0;i<n;i++){
    int num;
    cin >> num;
    q.push(num);
}
while(!q.empty()){
    res.push_back(q.front());
    q.pop();
    if(!q.empty()){
        q.push(q.front());
        q.pop();
    }
}
```

#### 栈

特点是后进先出，LIFO，入栈是将元素压入栈顶，出栈是将元素从栈顶弹出。

P2105表达式括号匹配

```cpp
从左到右扫描表达式，遇到左括号就将其压入栈中，遇到右括号就尝试从栈顶弹出一个左括号进行匹配。这样，最后一个压入栈的左括号（即最深层的左括号）将会是第一个被匹配的，这正好符合栈的后进先出特性。
while(true){
    char ch;
    cin >> ch;
    if(ch=='@'){
        break;
    }else if(ch=='('){
        st.push(ch);
    }else if(ch==')'){
        if(!st.empty()){
            st.pop();
        }else{
            cout << "NO" << endl;
            return 0;
        }
    }
}
//所以字符处理完，如果刚好匹配完的话，栈应该是空的
if(!st.empty()){
    cout << "NO" << endl;
    return 0;
}else{
    cout << "YES" << endl;
    return 0;
}
```

#### 集合

set可以去重+排序,插入是insert

P2103第k小的数  P2102随机数

```cpp
set<int> uniqueNumbers;
for(int i=0;i<n;i++){
    int num;
    cin >> num;
    uniqueNumbers.insert(num);
}
vector<int> sortedNumbers(uniqueNumbers.begin(),uniqueNumbers.end());
if(k<=sortedNumbers.size()){
    cout << sortedNumbers[k-1] << endl;
}else{
    cout << "NO RESULT" << endl;
}
```

#### 哈希

P3027无序数组中两数之和

```CPP
unordered_map<int,int> num_idx;
for(int i=0;i<n;i++){
    cin >> nums[i];
    num_idx[nums[i]] = i;
}
cin >> target;
for(int i=0;i<n;i++){
    auto it = num_idx.find(target-nums[i]);
    if(it != num_idx.end()){
        cout << i << " " << it->second << endl;
        return 0;
    }
    
}
```

#### 链表

STL中的list是一种双向链表，链表数据结构适合需要频繁在中间位置插入和删除元素的情况，比如P2104约瑟夫环也可以使用链表解决

```cpp
while(!l.empty()) {
    num = num % m + 1;
    if(num == m) {
        auto t = it;
        it++;
        printf("%d ", *t);
        l.erase(t);
    }
    else it++;
    if(it == l.end()) it = l.begin();
}
```

### 数论

P1002最大公约数

```cpp
int gcd(int a,int b){
    if(a==0) return b;
    return gcd(b%a,a);
}
```

P1004质数判断

```
遍历到sqrt(n)即可
```

P2009判断回文数字

```cpp
bool isRev(int x){
    if(x<0) return false;
    int original = x,reversed = 0;
    while(x){
        reversed = reversed * 10 + x%10;
        x/=10;
    }
    return reversed==original;
}
```

P2001列出所有约数

```cpp
遍历到n/2即可,最后输出n自身
```

P3002fib

```cpp
1.普通递归
int fibonacci_recursive(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2);
}
时间复杂度O(2^n)
    
2.递归优化——记录已经计算的值
int memo[1000] = {-1};  // 假设n的最大值为999，并初始化为-1表示未计算

int fibonacci_memo(int n) {
    if (n <= 1) {
        return n;
    }
    if (memo[n] != -1) {
        return memo[n];
    }
    memo[n] = fibonacci_memo(n-1) + fibonacci_memo(n-2);
    return memo[n];
}
时间复杂度O(n)
    
3.迭代
int fib(int n){
    if(n==0) return 0;
    if(n==1) return 1;
    int a=0,b=1,res;
    for(int i=2;i<=n;i++){
        res = a+b;
        a = b;
        b = res;
    }
    return res;
}
时间复杂度O(n)
```

### 前缀和与差分

前缀和与差分在处理数组或列表的累加或修改问题时非常有用。

**前缀和**：

前缀和主要用于处理数组区间的累加问题。前缀和的思想是预先计算出数组的累加和，然后在查询时，通过两次累加和的相减，快速得到一个区间的累加和。具体应用如例题P3025和为k的子数组

**差分**：

差分主要用于处理数组区间的修改问题。差分的思想是记录数组中每个元素与前一个元素的差值，然后在修改时，通过修改差分数组的两个值，快速实现一个区间的加减操作。例如，有一个数组nums，可以预先计算出一个差分数组diff，其中diff[i]表示nums[i] - nums[i-1]。那么，如果想要将nums数组中从下标i到下标j的元素都加上一个值v，只需要将diff[i]加上v，然后将diff[j+1]减去v即可。然后，可以通过差分数组重新计算出nums数组的值。具体应用如语文成绩。

#### 前缀和

P3025和为k的子数组

```
首先求得前缀和数组sum[i]
然后两层遍历寻找和为k的子数组的个数
for(int i=1;i<=n;i++){
    for(int j=0;j<i;j++){
        if(sum[i]-sum[j]==k){
            cnt++;
        }
    }
}
```

P5015石子合并

```
核心思想是简化运算,计算从第i堆到第j堆石子的总数量时，只需要计算前缀和数组的第j个元素减去第i-1个元素即可。
PS：前缀和数组的sum[0] = 0,便于处理边界问题
```

#### 差分

P3026语文成绩

```cpp
给x到y个学生增加z分，这种数组区间修改问题适用于使用差分数组：
1.构造差分数组
2.根据给定变化修改差分数组，比如c[x]+=z,合法的情况下c[y+1]-=z;
3.再根据差分数组重新计算出nums数组的值

//构造差分
for(int i=1;i<=n;i++){
    c[i] = nums[i]-nums[i-1];
}
while(p--){
    int x,y,z;
    cin >> x >> y >> z;
    c[x] += z;
    if(y+1<=n) c[y+1] -= z;
}
//回去
for(int i=1;i<=n;i++){
    nums[i] = nums[i-1] + c[i];
}
```

### 位运算

P2008二进制中1的个数

```cpp
int countOnes(int n){
	if(n==0) return 0;
	if(n==1) return 1;
	int cnt = 0;
	while(n){
		n &= (n-1);
		cnt++;
	}
	return cnt;
}
```

P6004快速幂

```cpp
ll quickmod(int a,int b,int c){
	int ans = 1;
	if(b==0) return 1%c;
	while(b){
        if(b&1) ans = (ll) ans * a % c;
		a = a * a % c;
		b>>=1;
    }
    return ans;
}
```

交换数字

```cpp
void swapNumbers(int &a, int &b) {
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}
```

判断是否为2的幂

```cpp
bool isPowerOfTwo(int n){
	return n>0 && n&(n-1)==0;
}
```

### 排序算法综合

冒泡排序（Bubble Sort）：通过比较相邻元素并交换不符合排序顺序的元素，使得每一轮循环后最大（或最小）的元素能够移动到序列的末尾。

选择排序（Selection Sort）：每一轮从待排序的元素中选出最小（或最大）的一个元素，存放在序列的起始位置，直到所有元素排序完毕。在未排序序列中找到最小（大）元素，存放到排序序列的起始位置从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。重复第二步，直到所有元素均排序完毕

插入排序（Insertion Sort）：通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。把待排序的数组分成已排序和未排序两部分，初始的时候把第一个元素认为是已排好序的从第二个元素开始，在已排好序的子数组中寻找到该元素合适的位置并插入该位置（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）重复上述过程直到最后一个元素被插入有序子数组中，表现最稳定的算法之一，无论什么数据进去都是O(n^2)的时间复杂度

快速排序（Quick Sort）：选择一个基准元素，通过一趟排序将待排序的数据分割成独立的两部分，其中一部分的所有数据都比另一部分的所有数据要小，然后再按此方法对这两部分数据分别进行快速排序。稳定，但代价是需要额外的内存空间。

归并排序（Merge Sort）：将已有序的子序列合并，得到完全有序的序列。先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为二路归并。

堆排序（Heap Sort）：利用堆这种数据结构所设计的一种排序算法。堆排序是一种选择排序，它的最坏，最好，平均时间复杂度均为O(nlogn)，它也是不稳定排序。

希尔排序（Shell Sort）：也称递减增量排序算法，是插入排序的一种更高效的改进版本。

计数排序（Counting Sort）、桶排序（Bucket Sort）和基数排序（Radix Sort）：这些都是非比较排序，适用于特定的数据范围。

### 时间复杂度分析

时间复杂度分析统计的不是算法运行时间，**而是算法运行时间随着数据量变大时的增长趋势**。时间复杂度相同，运行时间差别也可能很大。最差、最优时间复杂度会取决于数据分布，平均时间复杂度一般计算比较困难，所以一般使用最差时间复杂度。

注意： 
$$
n>=4时，n!>2^n
$$
<img src="E:/Typora%20Image/time_complexity_common_types-1704868625787-2.png" alt="常见的时间复杂度类型" style="zoom:50%;" />

**常数阶**

与输入数据大小无关。

```cpp
/* 常数阶 */
int constant(int n) {
    int count = 0;
    int size = 100000;
    for (int i = 0; i < size; i++)
        count++;
    return count;
}
```

**线性阶**

```cpp
一层循环，和输入数据大小呈线性关系
/* 线性阶 */
int linear(int n) {
    int count = 0;
    for (int i = 0; i < n; i++)
        count++;
    return count;
}
```

**对数阶**

```cpp
//代表性
public static void print2(int n){
    int i=1;
    while (i <= n) {
        i = i * 2;
    }
}
和指数相对
每一轮规模减半
/* 对数阶（递归实现） */
int logRecur(float n) {
    if (n <= 1)
        return 0;
    return logRecur(n / 2) + 1;
}
```

**平方阶**

```cpp
比如冒泡排序：
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {     
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}
```

**指数阶**

细胞分裂式的  比如fib    tower  递归分解为两个子问题，不优化的话就是指数

```cpp
/* 指数阶（循环实现） */
int exponential(int n) {
    int count = 0, base = 1;
    // 细胞每轮一分为二，形成数列 1, 2, 4, 8, ..., 2^(n-1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < base; j++) {
            count++;
        }
        base *= 2;
    }
    //外层 0 1 2 3 4 5 
    //内层 0 1 2 4 8 16 32...
    // count = 1 + 2 + 4 + 8 + .. + 2^(n-1) = 2^n - 1
    return count;
}
```

数据规模比较大的时候，使用dp或者贪心算法求解。

**阶乘阶**

```
/* 阶乘阶（递归实现） */
int factorialRecur(int n) {
    if (n == 0)
        return 1;
    int count = 0;
    // 从 1 个分裂出 n 个
    for (int i = 0; i < n; i++) {
        count += factorialRecur(n - 1);
    }
    return count;
}
PS:全排列问题时间复杂度是O(n!)
```




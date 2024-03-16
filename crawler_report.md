

**目录**

[toc]



# 爬虫

## 1 导入需要用到的包

### 1.1 用于获取网页的HTML内容并解析

```py
import requests
from bs4 import BeautifulSoup
```

### 1.2 用于合并URL

```py
from urllib.parse import urljoin
```

### 1.3 用于保存抓取的数据

```py
import csv
```

### 1.4 用于处理警告信息

```py
import warnings
warnings.filterwarnings('ignore')
```

## 2 爬虫

### 2.1 定义爬虫类

```py
class Crawler:
    def __init__(self, base_url, paths, headers):
        self.base_url = base_url
        self.paths = paths
        self.headers = headers
```

### 2.2 用于获取指定URL的正文内容

```py
    def get_web_content(self, url):
        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        article_div = soup.find('div', {'class': 'article typo'})
        if article_div:
            return article_div.text.strip()
        else:
            return self.get_wechat_content(url)
```

### 2.3 用于获取指定微信链接的正文内容

由于北京大学校友网中部分URL来源于微信公众号，故定义获取微信链接正文内容的函数 `get_wechat_content`

```py
    def get_wechat_content(self, url):
        response = requests.get(url, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        js_content = soup.find(id='js_content')
        if js_content:
            text = js_content.get_text().replace('\xa0','')
            text = text.split('责编：')[0]
            return text
        else:
            return None
```

### 2.4 用于获取所有爬取网页的标题、链接、日期和正文

通过`get_content`函数获取北京大学校友网—新闻—人物动态，北京大学校友网—人物—人物访谈、大师风范、学子风采、行业精英、人在海外、燕园学人共七个模块网页的标题、链接、日期、正文内容并写入csv文件`all_output.csv`  ，其中对于下一页的处理通过递归完成：如果在网页中找到了 "下一页" 的链接，那么获取下一页的 URL，并再次调用`get_content` 函数，抓取下一页的内容，直到没有下一页为止。

```py
    def get_content(self, url, base_url):
        print(f"Fetching {url}")  # 打印正在抓取的URL

        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'main_list_table'})
        rows = table.find_all('tr')[1:]  # skip the header row

        print(f"Found {len(rows)} rows")  # 打印找到的内容数量

        with open('all_output.csv', 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'link', 'date','article'])

            # 如果文件为空，写入表头
            if f.tell() == 0:
                writer.writeheader()

            for row in rows:
                cols = row.find_all('td')
                link = urljoin(base_url, cols[0].find('a')['href'])
                content = {
                    'head': cols[0].text.strip(),
                    'link': link,
                    'date': cols[1].text.strip(),
                    'content': self.get_web_content(link)
                }
                writer.writerow(content)

        next_pages = soup.find('div', {'class': 'page_info'}).find_all('a')
        if next_pages and "下一页" in next_pages[-1].text:
            next_page = next_pages[-1]
            next_page_url = urljoin(url, next_page['href'])
            print(next_page_url)
            self.get_content(next_page_url, base_url)
```

### 2.5 用于启动网页抓取过程

遍历所有传入的路径，通过 `base_url` 和 `path` 构造出完整的 URL，然后调用 `get_content` 方法抓取该 URL 的内容。

```py
    def run(self):
        for path in paths:
            url = base_url + path + 'index.htm'
            self.get_content(url, self.base_url + path)
```

运行过程中部分输出结果如图

<img src="E:/Typora%20Image/image-20231128234101268.png" alt="image-20231128234101268" style="zoom:50%;" />

### 2.6 主函数

首先定义 `base_url`、`paths` 和 `headers`变量；其次，使用这些变量创建一个 `Crawler` 对象；最后，调用 `Crawler` 对象的 `run` 方法，开始执行网页抓取的过程。

```py
if __name__ == '__main__':
    base_url = 'https://www.pku.org.cn/'
    paths = ['news/rwdt/','people/rwft/','people/dsff/','people/xzfc/','people/xyjy/','people/rzhw/','people/yyxr/']
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Cookies":"Hm_lvt_afc6a334e82d0ee23d872d2272de1389=1700755321,1700803470,1700804815,1700979864; Hm_lpvt_afc6a334e82d0ee23d872d2272de1389=1700980279"
    }
    crawler = Crawler(base_url, paths, headers)
    crawler.run()
```

部分爬取结果如图

![image-20231128234521653](E:/Typora%20Image/image-20231128234521653.png)

## 附：完整代码

```py
class Crawler:
    def __init__(self, base_url, paths, headers):
        self.base_url = base_url
        self.paths = paths
        self.headers = headers
	
    #定义获取微信链接正文内容的函数
    def get_wechat_content(self, url):
        response = requests.get(url, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        js_content = soup.find(id='js_content')
        if js_content:
            text = js_content.get_text().replace('\xa0','')
            text = text.split('责编：')[0]
            return text
        else:
            return None
	
    #定义获取网页正文内容的函数
    def get_web_content(self, url):
        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        article_div = soup.find('div', {'class': 'article typo'})
        if article_div:
            return article_div.text.strip()
        else:
            return self.get_wechat_content(url)
	
    #定义获取标题、链接、日期、正文内容的函数
    def get_content(self, url, base_url):
        print(f"Fetching {url}")  # 打印正在抓取的URL

        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'main_list_table'})
        rows = table.find_all('tr')[1:]  # skip the header row

        print(f"Found {len(rows)} rows")  # 打印找到的内容数量

        with open('all_output.csv', 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'link', 'date','article'])

            # 如果文件为空，写入表头
            if f.tell() == 0:
                writer.writeheader()

            for row in rows:
                cols = row.find_all('td')
                link = urljoin(base_url, cols[0].find('a')['href'])
                content = {
                    'head': cols[0].text.strip(),
                    'link': link,
                    'date': cols[1].text.strip(),
                    'content': self.get_web_content(link)
                }
                writer.writerow(content)
		
        #递归调用get_content函数获取下一页内容
        next_pages = soup.find('div', {'class': 'page_info'}).find_all('a')
        if next_pages and "下一页" in next_pages[-1].text:
            next_page = next_pages[-1]
            next_page_url = urljoin(url, next_page['href'])
            print(next_page_url)
            self.get_content(next_page_url, base_url)
	
    #定义Crawler类的主要入口点，启动网页抓取过程
    def run(self):
        for path in paths:
            url = base_url + path + 'index.htm'
            self.get_content(url, self.base_url + path)
```





# 分词+词云

## 1 文件夹结构

### 1.1 运行前

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282059508.png" alt="image-20231128205942458" style="zoom:50%;" />

### 1.2 运行后

关键词提取结果(result.xlsx)和词云图文件都保存在第一级文件夹下

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282100701.png" alt="image-20231128210017608" style="zoom:50%;" />



## 2 导入需要用到的包

### 2.1 用于数据处理和读写

```python
import pandas as pd
import openpyxl
```

### 2.2 用于文本分析

```python
import re
import jieba
import jieba.analyse as analyse
from collections import Counter
```

### 2.3 用于绘图和呈现

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba.posseg as pseg
```

## 3 关键词提取

### 3.1 读入爬虫结果csv文件

```python
filename = './all_output.csv'
df = pd.read_csv(filename,header=None)
df.columns = ['head', 'link', 'date','content']
```

将爬虫结果(一个无头csv)存储在一个df里,手动设置列名

### 3.2 关键词提取

思路:

采用jieba.analyse库实现TF-IDF算法关键词提取.

设置停用词库(来自网络,并添加词汇:['北大','北京大学','中国','年','月','日']使结果更美观)和词频库(来自jieba官方),自行提取每篇文章的15个关键词并写入爬虫结果文件

与此同时提取每篇文章的50个关键词,加入一个大列表tags,然后再用counter库统计大列表中的词频.

#### 设置停用词和词频库

```python
stopwords_path='./resource/cn_stopwords.txt'
idf_path='./resource/idf.txt.big'

sw = analyse.set_stop_words(stopwords_path)
idf = analyse.set_idf_path(idf_path)
```

#### 设置存储列表

```python
tags=[]#用于整体统计
taglst=[]#用于单篇文章
```

#### 提取关键词

遍历df的content列.

先用`re.sub(r'[^\u4e00-\u9fa5]', '', x)`过滤掉所有非文字符号.

再用`analyse.extract_tags`方法提取关键词,分别以\[[],[],[],[]...](通过append)和\[](通过+=)形式存入taglst和tags容器.

```python
for x in df['content']:
	try:
		text = re.sub(r'[^\u4e00-\u9fa5]', '', x)#用正则替换掉非文字符号
		tags+=analyse.extract_tags(text, topK=50, withWeight=False, allowPOS=(), withFlag=False)#用于整体词云图
		taglst.append(analyse.extract_tags(text, topK=15, withWeight=False, allowPOS=(), withFlag=False))#用于文章关键词
	except Exception as e:
		print(e)
		pass
```

#### 将单个文章的top15关键词写入excel文件

文件名为result.xlsx

```python
df['keywords']=taglst
print(df.head())

writer = pd.ExcelWriter('./result.xlsx', engine='openpyxl')
writer.book.create_sheet('Sheet1', 0)
df.to_excel(writer, sheet_name='Sheet1', index=True)
writer.close()
print("successfully written in excel")
```

效果如下:

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282042233.png" alt="image-20231128204230050" style="zoom:40%;" />



### 3.3 词频统计

#### 统计所有文章的总体词频

并显示排名前10的词频

```python
ct=Counter(tags)

top_words = ct.most_common(10)
print("\n频率最高的10个单词:")
for word, count in top_words:
    print(f"{word}: {count}")
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282021835.png" alt="image-20231128202149611" style="zoom:50%;" />







## 4 词性标注

思路:

将之前得到的所有文章top50关键词组成的大列表,用jieba.posseg进行词性标注,结果保存在一个df中.然后再根据词性筛选.

### 4.1 创建df

```python
df = pd.DataFrame(list(ct.items()), columns=['word', 'freq'])
df['flag'] = ''
```
### 4.2 标注词性

```python
for i, row in df.iterrows():
    words = pseg.cut(row['word'])
    pos_tags = [word.flag for word in words]
    df.at[i, 'flag'] = ' '.join(pos_tags)
```

### 4.3 分词性筛选df

```python
verb_df = df[df['flag'].str.contains('v')]
noun_df =df[df['flag'].str.contains('n')]
ad_df =df[df['flag'].str.contains('a|d')]
per_df =df[df['flag'].str.contains('nr')]
loc_df =df[df['flag'].str.contains('ns')]
```

词性对照表:

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282043070.jpg" alt="Xnip2023-11-28_20-05-37" style="zoom: 33%;" />



## 5 绘制词云图

### 5.1 词云图基本显示设置

```python
font_path='./resource/Songti.ttc'

wc = WordCloud(font_path=font_path, width=1500, height=1200, background_color='white',
               max_words=500, colormap='Dark2', contour_width=1, contour_color='black')
```

### 5.2 总词云

```python
wc.generate_from_frequencies(ct)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282036165.jpg" alt="wordcloud_verb" style="zoom:40%;" />

### 5.3 动词词云

```python
wcv = wc.generate_from_frequencies(dict(zip(verb_df['word'], verb_df['freq'])))
plt.imshow(wcv, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_verb.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282052801.jpg" alt="wordcloud_verb" style="zoom:40%;" />

### 5.4 名词词云

```python
wcn = wc.generate_from_frequencies(dict(zip(noun_df['word'], noun_df['freq'])))
plt.imshow(wcn, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_noun.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282037115.jpg" alt="wordcloud_noun" style="zoom:40%;" />



### 5.5 形容词和副词词云

```python
wcad = wc.generate_from_frequencies(dict(zip(ad_df['word'], ad_df['freq'])))
plt.imshow(wcad, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_ad.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282037680.jpg" alt="wordcloud_ad" style="zoom:40%;" />

### 5.6 人名词云

```python
wcp = wc.generate_from_frequencies(dict(zip(per_df['word'], per_df['freq'])))
plt.imshow(wcp, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_per.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282037580.jpg" alt="wordcloud_per" style="zoom:40%;" />

### 5.7 地名词云

```python
wcloc = wc.generate_from_frequencies(dict(zip(loc_df['word'], loc_df['freq'])))
plt.imshow(wcloc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_loc.jpg')
```

<img src="https://yytyyf.oss-cn-beijing.aliyuncs.com/test/202311282037828.jpg" alt="wordcloud_loc" style="zoom:40%;" />





## 附:完整代码

```python
import pandas as pd
import openpyxl
import re
import jieba
import jieba.analyse as analyse
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt



#读取csv数据,存入df
######################################################################################

# pd.set_option('display.unicode.east_asian_width',True)

filename = './all_output.csv'
df = pd.read_csv(filename,header=None)
df.columns = ['head', 'link', 'date','content']

print(df)

#关键词提取
######################################################################################

#给jieba.analyse设置停用词和词频库
stopwords_path='./resource/cn_stopwords.txt'
idf_path='./resource/idf.txt.big'

sw = analyse.set_stop_words(stopwords_path)
idf = analyse.set_idf_path(idf_path)


#统计开始

tags=[]#用于整体统计
taglst=[]#用于单篇文章

for x in df['content']:
	try:
		text = re.sub(r'[^\u4e00-\u9fa5]', '', x)#用正则替换掉非文字符号
		tags+=analyse.extract_tags(text, topK=50, withWeight=False, allowPOS=(), withFlag=False)#用于整体词云图
		taglst.append(analyse.extract_tags(text, topK=15, withWeight=False, allowPOS=(), withFlag=False))#用于文章关键词
	except Exception as e:
		print(e)
		pass

#总词频统计
ct=Counter(tags)

top_words = ct.most_common(10)
print("\n频率最高的10个单词:")
for word, count in top_words:
    print(f"{word}: {count}")



#文章关键词写入excel表格
######################################################################################

df['keywords']=taglst
print(df.head())

writer = pd.ExcelWriter('./result.xlsx', engine='openpyxl')
writer.book.create_sheet('Sheet1', 0)
df.to_excel(writer, sheet_name='Sheet1', index=True)
writer.close()#不能用.save(),这个方法已经失效了.
print("successfully written in excel")




#词性标注
######################################################################################

#创建df保存词性信息
df = pd.DataFrame(list(ct.items()), columns=['word', 'freq'])
df['flag'] = ''
for i, row in df.iterrows():
    words = pseg.cut(row['word'])
    pos_tags = [word.flag for word in words]
    df.at[i, 'flag'] = ' '.join(pos_tags)

#分词性筛选df
verb_df = df[df['flag'].str.contains('v')]
noun_df =df[df['flag'].str.contains('n')]
ad_df =df[df['flag'].str.contains('a|d')]
per_df =df[df['flag'].str.contains('nr')]
loc_df =df[df['flag'].str.contains('ns')]



#绘制词云图
######################################################################################

#词云图基本设置

font_path='./resource/Songti.ttc'

wc = WordCloud(font_path=font_path, width=1500, height=1200, background_color='white',
               max_words=500, colormap='Dark2', contour_width=1, contour_color='black')

#总词云

wc.generate_from_frequencies(ct)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud.jpg')


#动词词云

wcv = wc.generate_from_frequencies(dict(zip(verb_df['word'], verb_df['freq'])))
plt.imshow(wcv, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_verb.jpg')

#名词词云
wcn = wc.generate_from_frequencies(dict(zip(noun_df['word'], noun_df['freq'])))
plt.imshow(wcn, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_noun.jpg')


#形容词和副词词云
wcad = wc.generate_from_frequencies(dict(zip(ad_df['word'], ad_df['freq'])))
plt.imshow(wcad, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_ad.jpg')

#人名词云
wcp = wc.generate_from_frequencies(dict(zip(per_df['word'], per_df['freq'])))
plt.imshow(wcp, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_per.jpg')


#地名词云
wcloc = wc.generate_from_frequencies(dict(zip(loc_df['word'], loc_df['freq'])))
plt.imshow(wcloc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('./wordcloud_loc.jpg')
```


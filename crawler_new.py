#用于发送HTTP请求并获取响应
import requests

#用于解析HTML内容
from bs4 import BeautifulSoup

#用于拼接URL
from urllib.parse import urljoin

#用于将爬取结果写入CSV文件
import csv

#用于检查磁盘空间是否足够写入爬取结果
import shutil

#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

#用于防止过于频繁的请求被服务器封锁
import time

#用于记录日志
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Crawler:
    def __init__(self, base_url, paths, headers):
        self.base_url = base_url
        self.paths = paths
        self.headers = headers

    def get_wechat_content(self, url):
        response = requests.get(url, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        js_content = soup.find(id='js_content')
        if js_content:
            text = js_content.get_text().replace('\xa0','')
            text = text.split('责编：')[0]#去除“责编：”后面的内容
            return text
        else:
            return None

    def get_web_content(self, url):
        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser') 
        article_div = soup.find('div', {'class': 'article typo'})
        if article_div:
            return article_div.text.strip()
        else:
            return self.get_wechat_content(url)

    def get_content(self, url, base_url):
        max_retries = 2  # 最大重试次数
        retries = 0 # 当前重试次数
        while retries < max_retries:
            try:
                print(f"Fetching {url}")  # 打印正在抓取的URL

                # 发送HTTP请求，并设置超时时间
                response = requests.get(url, headers=self.headers, timeout=10, verify=False)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')

                table = soup.find('table', {'class': 'main_list_table'})
                rows = table.find_all('tr')[1:]  

                print(f"Found {len(rows)} rows")  # 打印找到的内容数量

                # 检查磁盘空间是否足够
                total, used, free = shutil.disk_usage("/")
                if free < 1024 * 1024 * 1:  # 例如，检查是否有至少10MB的空闲空间
                    print("Not enough disk space")
                    return
                
                # 将爬取结果写入CSV文件
                with open('tst1_output.csv', 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['head', 'link', 'date','content'])

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
                    url = urljoin(url, next_page['href'])   
                else:
                    url = None
                    
            #捕获requests异常
            except requests.exceptions.RequestException as e:
                logger.error(f"An error occurred while sending HTTP request: {e}. Retrying...")
                retries += 1
            #捕获非requests异常
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}. Retrying...")
                retries += 1
        if retries == max_retries:
            logger.error(f"Failed to fetch {url} after {retries} retries")

    def run(self):
        page_count = 0
        for path in self.paths:
            url = urljoin(self.base_url, path, 'index.htm')
            while url:
                self.get_content(url, urljoin(self.base_url, path))
                page_count += 1
                if page_count % 100 == 0:  # 每抓取100个网页后暂停
                    logger.info(f"Crawled {page_count} pages, sleeping for 2 seconds...")
                    time.sleep(2)  # 暂停2秒


if __name__ == '__main__':
    base_url = 'https://www.pku.org.cn/'
    paths = ['news/rwdt/','people/rwft/','people/dsff/','people/xzfc/','people/xyjy/','people/rzhw/','people/yyxr/']
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Cookies":"Hm_lvt_afc6a334e82d0ee23d872d2272de1389=1700755321,1700803470,1700804815,1700979864; Hm_lpvt_afc6a334e82d0ee23d872d2272de1389=1700980279"
    }
    crawler = Crawler(base_url, paths, headers)
    crawler.run()


from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.keys import Keys
import httpx
import json
import time
import logging
from typing import Optional, Dict, List, Any, Tuple
import traceback
import os
import sys
import cv2
import numpy as np
from PIL import Image
import io
import base64
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import random

def check_url(url: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    使用httpx检查URL的可访问性
    
    Args:
        url: 要检查的URL
    
    Returns:
        Tuple[bool, Optional[str], Optional[Dict]]: 
        - 是否可访问
        - 错误信息（如果有）
        - 响应头信息（如果成功）
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=10.0) as client:
            response = client.head(url)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return False, f"不支持的内容类型: {content_type}", None
                
            return True, None, dict(response.headers)
            
    except httpx.HTTPStatusError as e:
        return False, f"HTTP错误: {e.response.status_code}", None
    except httpx.RequestError as e:
        return False, f"请求错误: {str(e)}", None
    except Exception as e:
        return False, f"未知错误: {str(e)}", None

class DynamicCrawler:
    """动态网页爬虫类"""
    
    def __init__(self, headless: bool = True, retry_times: int = 3):
        """
        初始化动态爬虫
        
        Args:
            headless: 是否使用无头模式
            retry_times: 操作失败时的重试次数
        """
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        
        self.retry_times = retry_times
        self.session = httpx.Client(follow_redirects=True)
        
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59")
        
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)
        
        self.driver = None
        self.wait = None
        self.actions = None
        
    def start(self):
        try:
            service = Service(EdgeChromiumDriverManager().install())
            self.driver = webdriver.Edge(service=service, options=self.options)
            
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    """
                }
            )
            
            self.wait = WebDriverWait(self.driver, 10)
            self.actions = ActionChains(self.driver)
            logging.info("浏览器启动成功")
            return True
        except Exception as e:
            logging.error(f"浏览器启动失败: {e}")
            return False
            
    def close(self):
        """关闭浏览器和会话"""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("浏览器已关闭")
            except Exception as e:
                logging.error(f"关闭浏览器时出错: {e}")
        
        if self.session:
            try:
                self.session.close()
                logging.info("HTTP会话已关闭")
            except Exception as e:
                logging.error(f"关闭HTTP会话时出错: {e}")

    def wait_for_element(self, 
                        selector: str, 
                        timeout: int = 10, 
                        condition: str = "presence") -> Optional[Any]:
        """
        智能等待元素
        
        Args:
            selector: CSS选择器
            timeout: 超时时间（秒）
            condition: 等待条件（presence/visible/clickable）
            
        Returns:
            找到的元素或None
        """
        try:
            if condition == "presence":
                element = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
            elif condition == "visible":
                element = self.wait.until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
                )
            elif condition == "clickable":
                element = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
            else:
                raise ValueError(f"不支持的等待条件: {condition}")
                
            return element
        except TimeoutException:
            logging.error(f"等待元素超时: {selector}")
            return None
        except Exception as e:
            logging.error(f"等待元素时出错: {e}")
            return None

    def download_image(self, url_or_base64: str) -> Optional[np.ndarray]:
        """
        下载图片并转换为OpenCV格式
        
        Args:
            url_or_base64: 图片URL或Base64字符串
            
        Returns:
            OpenCV格式的图片数组或None
        """
        try:
            if url_or_base64.startswith('data:image'):
                # Base64格式
                img_data = base64.b64decode(url_or_base64.split(',')[1])
            else:
                # URL格式
                response = self.session.get(url_or_base64)
                img_data = response.content
                
            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return img
            
        except Exception as e:
            logging.error(f"下载或转换图片时出错: {e}")
            return None

    def retry_operation(self, operation, *args, **kwargs) -> Tuple[bool, Any]:
        """
        重试操作直到成功或达到最大重试次数
        
        Args:
            operation: 要重试的操作（函数）
            *args: 传递给操作的位置参数
            **kwargs: 传递给操作的关键字参数
            
        Returns:
            Tuple[是否成功, 操作结果]
        """
        for i in range(self.retry_times):
            try:
                result = operation(*args, **kwargs)
                return True, result
            except Exception as e:
                logging.warning(f"操作失败 (尝试 {i+1}/{self.retry_times}): {e}")
                if i < self.retry_times - 1:
                    time.sleep(1)  # 失败后等待1秒再重试
                continue
        return False, None
    
    def get_page(self, url: str, wait_for: str = "body") -> bool:
        try:
            self.driver.get(url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, wait_for)))
            return True
        except TimeoutException:
            logging.error(f"页面加载超时: {url}")
            return False
        except WebDriverException as e:
            logging.error(f"访问页面时出错: {e}")
            return False
    
    def extract_data(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        data = {}
        for name, selector in selectors.items():
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if len(elements) == 1:
                    data[name] = elements[0].text
                elif len(elements) > 1:
                    data[name] = [el.text for el in elements]
                else:
                    data[name] = None
            except Exception as e:
                logging.error(f"提取数据 '{name}' 时出错: {e}")
                data[name] = None
        return data
    
    def crawl_page(self, url: str) -> Dict[str, Any]:
        result = {"success": False, "data": None, "error": None}
        
        try:
            # 访问页面
            if not self.get_page(url):
                result["error"] = "页面加载失败"
                return result
            
            # 等待页面加载完成
            time.sleep(2)
            
            # 提取常见数据
            selectors = {
                "title": "title",  # 页面标题
                "h1": "h1",  # 主标题
                "h2": "h2",  # 副标题
                "paragraphs": "p",  # 段落
                "links": "a",  # 链接
                "images": "img",  # 图片
                "lists": "ul, ol",  # 列表
            }
            
            # 提取数据
            data = self.extract_data(selectors)
            
            # 获取页面源码
            data["html"] = self.driver.page_source
            
            # 获取页面URL（可能会因重定向而改变）
            data["url"] = self.driver.current_url
            
            result["success"] = True
            result["data"] = data
            
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"爬取过程中出错: {e}")
            traceback.print_exc()
            
        return result

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=== 动态网页爬虫 (Edge浏览器版) ===")
    url = input("请输入要爬取的网页URL: ").strip()
    
    if not url:
        print("错误: URL不能为空")
        return
        
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    print(f"正在检查URL可访问性: {url}")
    
    # 首先使用httpx检查URL
    accessible, error_msg, headers = check_url(url)
    if not accessible:
        print(f"URL检查失败: {error_msg}")
        return
        
    print("URL检查通过，开始爬取...")
    if headers:
        print(f"服务器类型: {headers.get('server', '未知')}")
        print(f"内容类型: {headers.get('content-type', '未知')}")
    
    # 创建爬虫实例
    crawler = DynamicCrawler(headless=False)
    
    try:
        # 启动浏览器
        if not crawler.start():
            print("Edge浏览器启动失败，请检查EdgeDriver配置")
            return
        
        # 爬取页面
        result = crawler.crawl_page(url)
        
        if result["success"]:
            print("\n爬取成功!")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"crawled_data_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result["data"], f, ensure_ascii=False, indent=2)
                
            print(f"数据已保存到: {filename}")
            
            # 显示部分数据
            print("\n页面标题:", result["data"].get("title"))
            print("\n主标题:", result["data"].get("h1"))
            
            # 显示段落数量
            paragraphs = result["data"].get("paragraphs")
            if paragraphs and isinstance(paragraphs, list):
                print(f"\n共找到 {len(paragraphs)} 个段落")
                
            # 显示链接数量
            links = result["data"].get("links")
            if links and isinstance(links, list):
                print(f"共找到 {len(links)} 个链接")
                
            # 显示图片数量
            images = result["data"].get("images")
            if images and isinstance(images, list):
                print(f"共找到 {len(images)} 张图片")
                
        else:
            print(f"爬取失败: {result['error']}")
            
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        traceback.print_exc()
    finally:
        # 关闭浏览器
        crawler.close()

if __name__ == "__main__":
    main()
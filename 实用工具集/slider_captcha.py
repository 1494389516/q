import cv2
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import base64
import time
import random
from typing import Tuple, List, Optional
import logging
from 动态爬虫 import DynamicCrawler

class SliderCaptchaCrawler(DynamicCrawler):
    """处理滑动验证码的爬虫类"""
    
    def __init__(self, headless: bool = False):
        # 滑动验证码不建议使用headless模式
        super().__init__(headless=headless)
        
    def get_slider_and_background(self, 
                                slider_selector: str,
                                background_selector: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取滑块和背景图片
        
        Args:
            slider_selector: 滑块图片元素的CSS选择器
            background_selector: 背景图片元素的CSS选择器
            
        Returns:
            Tuple[滑块图片ndarray, 背景图片ndarray]
        """
        try:
            # 等待图片元素加载
            slider_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, slider_selector))
            )
            background_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, background_selector))
            )
            
            # 获取图片src（可能是Base64或URL）
            slider_src = slider_element.get_attribute('src')
            background_src = background_element.get_attribute('src')
            
            # 转换为OpenCV格式
            slider_img = self._src_to_cv2(slider_src)
            background_img = self._src_to_cv2(background_src)
            
            return slider_img, background_img
            
        except TimeoutException:
            logging.error("等待验证码图片超时")
            return None, None
        except Exception as e:
            logging.error(f"获取验证码图片时出错: {e}")
            return None, None
            
    def _src_to_cv2(self, src: str) -> Optional[np.ndarray]:
        """
        将图片src转换为OpenCV格式
        
        Args:
            src: 图片源（Base64或URL）
            
        Returns:
            OpenCV格式的图片数组
        """
        try:
            if src.startswith('data:image'):
                # Base64格式
                img_data = base64.b64decode(src.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # URL格式
                resp = self.session.get(src)
                nparr = np.frombuffer(resp.content, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"图片转换失败: {e}")
            return None
            
    def find_slider_position(self, 
                           slider_img: np.ndarray, 
                           background_img: np.ndarray) -> Optional[int]:
        """
        使用模板匹配找到滑块的目标位置
        
        Args:
            slider_img: 滑块图片
            background_img: 背景图片
            
        Returns:
            滑块的目标x坐标
        """
        try:
            # 转换为灰度图
            slider_gray = cv2.cvtColor(slider_img, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            slider_edge = cv2.Canny(slider_gray, 100, 200)
            background_edge = cv2.Canny(background_gray, 100, 200)
            
            # 模板匹配
            result = cv2.matchTemplate(background_edge, slider_edge, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 返回最佳匹配位置的x坐标
            return max_loc[0]
            
        except Exception as e:
            logging.error(f"查找滑块位置时出错: {e}")
            return None
            
    def generate_slide_track(self, distance: int) -> List[int]:
        """
        生成人性化的滑动轨迹
        
        Args:
            distance: 需要滑动的总距离
            
        Returns:
            滑动轨迹列表，每个元素代表每次移动的距离
        """
        # 初始化轨迹列表
        tracks = []
        
        # 当前位置
        current = 0
        
        # 减速阈值
        mid = distance * 4 / 5
        
        # 计算间隔
        t = 0.2
        
        # 初速度
        v = 0
        
        while current < distance:
            if current < mid:
                # 加速度为正
                a = 2
            else:
                # 减速
                a = -3
            
            # 初速度v0
            v0 = v
            
            # 当前速度v = v0 + at
            v = v0 + a * t
            
            # 移动距离x = v0t + 1/2 * a * t^2
            move = v0 * t + 1/2 * a * t * t
            
            # 当前位置
            current += move
            
            # 加入轨迹
            tracks.append(round(move))
        
        # 微调，使得总和正好等于距离
        while sum(tracks) > distance:
            tracks[-1] -= 1
        while sum(tracks) < distance:
            tracks[-1] += 1
            
        return tracks
        
    def slide_captcha(self, 
                     slider_element_selector: str,
                     slider_img_selector: str,
                     background_img_selector: str) -> bool:
        """
        完整的滑动验证码破解流程
        
        Args:
            slider_element_selector: 滑块元素的CSS选择器
            slider_img_selector: 滑块图片的CSS选择器
            background_img_selector: 背景图片的CSS选择器
            
        Returns:
            是否成功破解
        """
        try:
            # 1. 获取图片
            slider_img, background_img = self.get_slider_and_background(
                slider_img_selector,
                background_img_selector
            )
            if slider_img is None or background_img is None:
                return False
                
            # 2. 计算需要滑动的距离
            target_x = self.find_slider_position(slider_img, background_img)
            if target_x is None:
                return False
                
            # 3. 生成滑动轨迹
            tracks = self.generate_slide_track(target_x)
            
            # 4. 获取滑块元素
            slider = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, slider_element_selector))
            )
            
            # 5. 按下滑块
            self.actions.click_and_hold(slider).perform()
            
            # 6. 按轨迹移动
            for track in tracks:
                self.actions.move_by_offset(track, random.randint(-2, 2)).perform()
                time.sleep(random.uniform(0.01, 0.03))
                
            # 7. 稳定一下
            time.sleep(0.5)
            
            # 8. 释放
            self.actions.release().perform()
            
            # 9. 等待验证结果
            time.sleep(2)
            
            return True
            
        except Exception as e:
            logging.error(f"滑动验证码破解失败: {e}")
            return False
            
def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建爬虫实例
    crawler = SliderCaptchaCrawler(headless=False)
    
    try:
        # 启动浏览器
        if not crawler.start():
            print("浏览器启动失败")
            return
            
        # 访问测试页面（这里需要替换为实际的验证码测试页面）
        url = input("请输入包含滑动验证码的测试页面URL: ")
        if not crawler.get_page(url):
            print("页面访问失败")
            return
            
        # 尝试破解验证码
        # 注意：这里的选择器需要根据实际页面来修改
        success = crawler.slide_captcha(
            slider_element_selector=".slider",  # 滑块元素
            slider_img_selector=".slider-img",  # 滑块图片
            background_img_selector=".background-img"  # 背景图片
        )
        
        if success:
            print("验证码破解成功！")
        else:
            print("验证码破解失败")
            
    except Exception as e:
        print(f"测试过程出错: {e}")
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
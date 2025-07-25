"""
验证码处理模块
"""

import base64
import io
import logging
from typing import Optional, Union
from PIL import Image, ImageFilter

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from twocaptcha import TwoCaptcha
    TWOCAPTCHA_AVAILABLE = True
except ImportError:
    TWOCAPTCHA_AVAILABLE = False


class CaptchaSolver:
    """验证码解决器"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # 设置Tesseract路径
        if TESSERACT_AVAILABLE and tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.tesseract_available = TESSERACT_AVAILABLE
        self.twocaptcha_available = TWOCAPTCHA_AVAILABLE
    
    def solve_image_captcha(self, image_data: Union[bytes, str, Image.Image]) -> Optional[str]:
        """解决图片验证码
        
        Args:
            image_data: 图片数据，可以是字节、base64字符串或PIL Image对象
            
        Returns:
            识别出的验证码文本
        """
        if not self.tesseract_available:
            self.logger.warning("Tesseract不可用，无法进行本地验证码识别")
            return None
        
        try:
            # 转换为PIL Image对象
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # base64格式
                    b64_data = image_data.split(',')[1]
                    img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                else:
                    # 文件路径
                    img = Image.open(image_data)
            else:
                img = image_data
            
            # 图片预处理
            img = self._preprocess_image(img)
            
            # OCR识别
            result = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            return result.strip()
        
        except Exception as e:
            self.logger.error(f"图片验证码识别失败: {e}")
            return None
    
    def solve_slider_captcha(self, bg_image: Union[bytes, str, Image.Image], 
                           slider_image: Union[bytes, str, Image.Image]) -> Optional[int]:
        """解决滑块验证码
        
        Args:
            bg_image: 背景图片
            slider_image: 滑块图片
            
        Returns:
            滑块应该移动的距离（像素）
        """
        try:
            # 转换为PIL Image对象
            bg = self._to_pil_image(bg_image)
            slider = self._to_pil_image(slider_image)
            
            # 转换为灰度图
            bg_gray = bg.convert('L')
            slider_gray = slider.convert('L')
            
            # 模板匹配
            bg_data = bg_gray.load()
            slider_data = slider_gray.load()
            
            min_diff = float('inf')
            best_x = 0
            
            # 在背景图上滑动滑块，寻找最佳匹配位置
            for x in range(bg.width - slider.width):
                diff_sum = 0
                for sy in range(slider.height):
                    for sx in range(slider.width):
                        if x + sx < bg.width and sy < bg.height:
                            diff = abs(int(bg_data[x + sx, sy]) - int(slider_data[sx, sy]))
                            diff_sum += diff
                
                if diff_sum < min_diff:
                    min_diff = diff_sum
                    best_x = x
            
            return best_x
        
        except Exception as e:
            self.logger.error(f"滑块验证码处理失败: {e}")
            return None
    
    def solve_with_2captcha(self, api_key: str, 
                           image_data: Optional[Union[bytes, str]] = None,
                           site_key: Optional[str] = None,
                           page_url: Optional[str] = None,
                           captcha_type: str = 'image') -> Optional[str]:
        """使用2captcha服务解决验证码
        
        Args:
            api_key: 2captcha API密钥
            image_data: 图片数据（用于图片验证码）
            site_key: 站点密钥（用于reCAPTCHA）
            page_url: 页面URL（用于reCAPTCHA）
            captcha_type: 验证码类型 ('image', 'recaptcha', 'hcaptcha')
            
        Returns:
            解决后的验证码结果
        """
        if not self.twocaptcha_available:
            self.logger.warning("2captcha库不可用")
            return None
        
        try:
            solver = TwoCaptcha(api_key)
            
            if captcha_type == 'image' and image_data:
                if isinstance(image_data, str) and image_data.startswith('data:image'):
                    # base64格式
                    response = solver.normal(image_data)
                elif isinstance(image_data, bytes):
                    # 保存为临时文件
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(image_data)
                        tmp_path = tmp_file.name
                    
                    try:
                        response = solver.normal(tmp_path)
                    finally:
                        os.unlink(tmp_path)
                else:
                    # 文件路径
                    response = solver.normal(image_data)
                
                return response['code']
            
            elif captcha_type == 'recaptcha' and site_key and page_url:
                response = solver.recaptcha(sitekey=site_key, url=page_url)
                return response['code']
            
            elif captcha_type == 'hcaptcha' and site_key and page_url:
                response = solver.hcaptcha(sitekey=site_key, url=page_url)
                return response['code']
            
            else:
                self.logger.error(f"不支持的验证码类型或缺少必要参数: {captcha_type}")
                return None
        
        except Exception as e:
            self.logger.error(f"2captcha服务解决验证码失败: {e}")
            return None
    
    def _to_pil_image(self, image_data: Union[bytes, str, Image.Image]) -> Image.Image:
        """转换为PIL Image对象"""
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            if image_data.startswith('data:image'):
                b64_data = image_data.split(',')[1]
                return Image.open(io.BytesIO(base64.b64decode(b64_data)))
            else:
                return Image.open(image_data)
        else:
            return image_data
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """图片预处理，提高OCR识别率"""
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 应用中值滤波去噪
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # 二值化
        threshold = 128
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
        
        # 可选：放大图片以提高识别率
        width, height = img.size
        if width < 200 or height < 50:
            scale_factor = max(200 / width, 50 / height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
        
        return img
    
    def get_capabilities(self) -> dict:
        """获取验证码解决器的能力"""
        return {
            'local_ocr': self.tesseract_available,
            'slider_captcha': True,
            '2captcha_service': self.twocaptcha_available,
            'supported_types': [
                'image_captcha',
                'slider_captcha',
                'recaptcha' if self.twocaptcha_available else None,
                'hcaptcha' if self.twocaptcha_available else None
            ]
        }
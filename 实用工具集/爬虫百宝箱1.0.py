import requests
import re
import os
import time
import random

def crawl_images_revised(page_url, html_content_for_images, headers_dict):
    processed_page_url = page_url
    if processed_page_url.startswith("http://"):
        processed_page_url = "https://" + processed_page_url[len("http://"):]
    elif not processed_page_url.startswith("https://") and "://" not in processed_page_url:
        processed_page_url = "https://" + processed_page_url

    image_pattern = r'<img.*?src=["\'](.*?)["\'].*?>'
    img_urls_found = re.findall(image_pattern, html_content_for_images, re.IGNORECASE)

    print("找到的图片URL (默认使用HTTPS):")
    processed_urls = set()

    base_domain_url = ""
    match_domain = re.match(r'(https?://[^/]+)', processed_page_url)
    if match_domain:
        base_domain_url = match_domain.group(1)
        if base_domain_url.startswith("http://"):
            base_domain_url = "https://" + base_domain_url[len("http://"):]
    else:
        domain_part = processed_page_url.split('/')[0]
        if "://" not in domain_part:
            base_domain_url = "https://" + domain_part
        else:
            print(f"警告: 无法从 {processed_page_url} 解析域名。根相对路径的图片可能无法正确处理。")

    current_path_base = processed_page_url
    if not current_path_base.endswith('/'):
        last_slash_idx = current_path_base.rfind('/')
        if last_slash_idx > current_path_base.find("://") + 2:
            current_path_base = current_path_base[:last_slash_idx + 1]
        else:
            current_path_base += '/'
            
    if current_path_base.startswith("http://"):
        current_path_base = "https://" + current_path_base[len("http://"):]
    elif not current_path_base.startswith("https://") and "://" not in current_path_base.split('/')[0]:
        current_path_base = "https://" + current_path_base

    for img_url in img_urls_found:
        final_url = ""
        original_img_url = img_url

        if img_url.startswith("data:"):
            continue
        elif img_url.startswith("//"):
            final_url = f"https:{img_url}"
        elif img_url.startswith("/"):
            if base_domain_url:
                final_url = f"{base_domain_url}{img_url}"
            else:
                final_url = img_url 
        elif img_url.startswith("http://"):
            final_url = "https://" + img_url[len("http://"):]
        elif img_url.startswith("https://"):
            final_url = img_url
        else:
            if current_path_base:
                final_url = current_path_base + img_url
                if final_url.startswith("http://"):
                     final_url = "https://" + final_url[len("http://"):]
            else:
                final_url = original_img_url

        if final_url and final_url not in processed_urls:
            if final_url.startswith("http://"):
                final_url = "https://" + final_url[len("http://"):]
            elif not final_url.startswith("https://") and not final_url.startswith("data:") and "://" not in final_url.split('/')[0]:
                 final_url = "https://" + final_url
            
            if final_url.startswith("https://"):
                 print(final_url)
                 processed_urls.add(final_url)
            elif not final_url.startswith("data:"):
                 print(f"提示: 未能解析为有效HTTPS的图片URL: {original_img_url} (尝试结果: {final_url})")

def crawl_comments(target_url, headers_dict):
    print(f"正在尝试从 {target_url} 爬取评论...")
    print("提示: 爬取评论功能需要针对特定网站结构进行定制开发。")
    print("VIP内容或动态加载的评论通常需要更复杂的抓取策略。")

def crawl_videos(target_url, headers_dict, html_content):
    print(f"正在尝试从 {target_url} 爬取视频到本地...")
    
    video_pattern = r'<video.*?src=["\'](.*?)["\'].*?>'
    video_urls = re.findall(video_pattern, html_content, re.IGNORECASE)
    
    source_pattern = r'<source.*?src=["\'](.*?)["\'].*?>'
    source_urls = re.findall(source_pattern, html_content, re.IGNORECASE)
    
    embed_pattern = r'<iframe.*?src=["\'](.*?)["\'].*?>'
    embed_urls = re.findall(embed_pattern, html_content, re.IGNORECASE)
    
    js_video_pattern = r'video[Uu]rl\s*[:=]\s*["\'](.*?)["\']'
    js_video_urls = re.findall(js_video_pattern, html_content)
    
    video_urls.extend(source_urls)
    
    all_urls = list(set(video_urls + embed_urls + js_video_urls))
    
    if not all_urls:
        print("未找到直接可见的视频链接，正在尝试分析页面中的播放器...")
        print("提示：许多网站使用加密方式或动态加载视频，尤其是VIP内容")
        
        try:
            referer_headers = headers_dict.copy()
            referer_headers['Referer'] = target_url
            
            print("正在尝试检测可能的m3u8播放列表...")
            m3u8_pattern = r'["\']([^"\']*?\.m3u8[^"\']*?)["\']'
            m3u8_urls = re.findall(m3u8_pattern, html_content)
            
            for m3u8_url in m3u8_urls:
                if m3u8_url.startswith("//"):
                    m3u8_url = f"https:{m3u8_url}"
                elif m3u8_url.startswith("/"):
                    base_url_match = re.match(r'(https?://[^/]+)', target_url)
                    if base_url_match:
                        base_url = base_url_match.group(1)
                        m3u8_url = f"{base_url}{m3u8_url}"
                elif not m3u8_url.startswith(("http://", "https://")):
                    base_url_match = re.match(r'(https?://[^/]+/[^/]+/)', target_url)
                    if base_url_match:
                        base_path = base_url_match.group(1)
                        m3u8_url = f"{base_path}{m3u8_url}"
                
                if m3u8_url.startswith("http://"):
                    m3u8_url = "https://" + m3u8_url[len("http://"):]
                
                print(f"发现可能的m3u8播放列表: {m3u8_url}")
                all_urls.append(m3u8_url)
        except Exception as e:
            print(f"分析播放列表时出错: {e}")
    
    if not all_urls:
        print("未找到视频链接。这可能是因为:")
        print("1. 视频内容被加密或需要特殊播放器")
        print("2. 该页面可能需要VIP权限才能访问视频内容")
        print("3. 视频内容通过JavaScript动态加载")
        return
    
    print(f"\n找到 {len(all_urls)} 个可能的视频相关链接")
    
    download_dir = "downloaded_videos"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for i, video_url in enumerate(all_urls):
        processed_url = video_url
        if processed_url.startswith("//"):
            processed_url = f"https:{processed_url}"
        elif processed_url.startswith("/"):
            base_url_match = re.match(r'(https?://[^/]+)', target_url)
            if base_url_match:
                base_url = base_url_match.group(1)
                processed_url = f"{base_url}{processed_url}"
        elif processed_url.startswith("http://"):
            processed_url = "https://" + processed_url[len("http://"):]
            
        print(f"\n正在处理 #{i+1}: {processed_url}")
        
        if ".m3u8" in processed_url:
            print("检测到m3u8播放列表，这通常是流媒体内容")
            print("提示: 流媒体内容通常需要特殊工具如ffmpeg进行下载")
            print("该链接可能是视频的加密播放列表，可能需要额外的解密步骤")
            continue
            
        if "iframe" in processed_url or "embed" in processed_url:
            print("这是嵌入式播放器链接，可能需要进一步分析")
            continue
        
        try:
            file_name = f"{download_dir}/video_{i+1}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            extension = ""
            if '.' in processed_url.split('/')[-1]:
                extension = '.' + processed_url.split('/')[-1].split('.')[-1]
                if len(extension) > 5 or extension == '.html' or extension == '.php':
                    extension = '.mp4'
            else:
                extension = '.mp4'
                
            file_name += extension
            
            print(f"正在尝试下载到: {file_name}")
            
            download_headers = headers_dict.copy()
            download_headers['Referer'] = target_url
            
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    response = requests.get(processed_url, headers=download_headers, stream=True, timeout=15)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'text/html' in content_type.lower():
                        print(f"警告: 返回的是HTML内容而非视频文件")
                        break
                        
                    if not ('video' in content_type.lower() or 'stream' in content_type.lower() or 'application/' in content_type.lower()):
                        print(f"警告: 内容类型 {content_type} 可能不是视频")
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    if total_size < 10000:  # 小于10KB可能不是视频文件
                        print(f"警告: 文件大小只有 {total_size} 字节，可能不是视频文件")
                        
                    with open(file_name, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                    
                    print(f"下载完成: {file_name}")
                    break
                except Exception as e:
                    print(f"尝试 {attempt+1}/{retry_count} 失败: {e}")
                    if attempt < retry_count - 1:
                        time.sleep(2)  # 等待2秒后重试
                    else:
                        print("下载失败，已达到最大重试次数")
        except Exception as e:
            print(f"处理视频URL时出错: {e}")
    
    print(f"\n视频爬取任务完成。下载的文件保存在 {os.path.abspath(download_dir)} 目录中")
    print("注意: 某些下载内容可能不完整或需要进一步处理才能播放")

def main():
    print("欢迎使用爬虫百宝箱")
    print("请注意：爬取受登录保护或有复杂反爬机制（如VIP内容）的网站可能失败。")
    print("请确保您的爬取行为符合目标网站的服务条款和相关法律法规。")
    print("\n请输入请求头信息 (Headers):")
    custom_headers = {}
    
    user_agent = input("User-Agent (必填): ").strip()
    if not user_agent:
        print("错误: User-Agent是必填项。程序将退出。")
        return
    custom_headers['User-Agent'] = user_agent
    
    target_url_input = input("\n请输入目标网址: ")
    
    processed_target_url = target_url_input.strip()
    if not processed_target_url:
        print("错误: 未输入目标网址。程序将退出。")
        return

    if not processed_target_url.startswith("http://") and not processed_target_url.startswith("https://"):
        if "://" in processed_target_url:
            print(f"错误: 不支持的URL协议。请输入以 http:// 或 https:// 开头的网址，或纯域名。")
            return
        processed_target_url = "https://" + processed_target_url
    elif processed_target_url.startswith("http://"):
        processed_target_url = "https://" + processed_target_url[len("http://"):]

    print(f"目标URL (将使用HTTPS协议): {processed_target_url}")
    
    print("\n请选择:")
    print("1. 爬取图片链接")
    print("2. 爬取评论 (占位功能)")
    print("3. 爬取视频到本地")
    choice = input("请输入选项 (1-3): ").strip()

    try:
        print(f"\n正在请求: {processed_target_url} ...")
        response = requests.get(processed_target_url, headers=custom_headers, timeout=10)
        response.raise_for_status() 
        response.encoding = response.apparent_encoding
        html_content_main = response.text
        print(f"成功获取页面内容。")
    except requests.exceptions.Timeout:
        print(f"请求超时: 无法在10秒内连接到 {processed_target_url}")
        return
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err} - {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
        return
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return

    if choice == '1':
        print("\n--- 开始爬取图片链接 ---")
        crawl_images_revised(processed_target_url, html_content_main, custom_headers)
    elif choice == '2':
        print("\n--- 开始爬取评论 ---")
        crawl_comments(processed_target_url, custom_headers)
    elif choice == '3':
        print("\n--- 开始爬取视频到本地 ---")
        crawl_videos(processed_target_url, custom_headers, html_content_main)
    else:
        print("无效的选项。程序将退出。")
    
    print("\n--- 爬取任务结束 ---")

if __name__ == "__main__":
    main()







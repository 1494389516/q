import requests
import re
import os
import time
import random
from bs4 import BeautifulSoup
import shutil
from urllib.parse import urlparse
import xlwt
import traceback

DEFAULT_TIMEOUT = 30
COMMENT_TIMEOUT = 10

def douban_spider(url):
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None

def parse_bilibili_video(bvid, referer, headers):
    try:
        api_url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&qn=116&fnval=16"
        headers = {
            **headers,
            'Referer': referer,
            'Origin': 'https://www.bilibili.com',
            'Accept': 'application/json, text/plain, */*'
        }
        response = requests.get(api_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data['code'] != 0:
            print(f"B站API错误: {data.get('message')}")
            return None
        return download_bilibili_m3u8(data, referer, headers)
    except requests.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"B站解析失败: {str(e)}")
    return None

def download_bilibili_m3u8(data, referer, headers):
    download_dir = "downloaded_videos"
    os.makedirs(download_dir, exist_ok=True)
    video_info = data['data']['dash']['video'][0]
    audio_info = data['data']['dash']['audio'][0]
    print(f"发现视频流: {video_info['id']} | 分辨率: {video_info['width']}x{video_info['height']}")
    video_url = video_info['baseUrl']
    audio_url = audio_info['baseUrl']
    temp_dir = f"temp_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)
    def download_stream(url, name):
        try:
            print(f"开始下载{name}流...")
            headers['Referer'] = referer
            response = requests.get(url, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT)
            with open(f"{temp_dir}/{name}.mp4", 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"{name}流下载完成")
        except Exception as e:
            print(f"{name}流下载失败:", str(e))
            return False
        return True
    if download_stream(video_url, 'video') and download_stream(audio_url, 'audio'):
        output_file = f"{download_dir}/bilibili_{int(time.time())}.mp4"
        try:
            print("开始合并音视频...")
            shutil.move(f"{temp_dir}/video.mp4", output_file)
            shutil.rmtree(temp_dir)
            print(f"视频已保存至: {output_file}")
        except Exception as e:
            print("合并失败:", str(e))

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
    try:
        response = requests.get(target_url, headers=headers_dict, timeout=COMMENT_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        comment_selectors = [
            '.comment-text', '.comment-body', '.comment-content', '.comment',
            'div[class*="comment"]', 'article[class*="comment"]',
            '.reply-content', '.post-text', '.comment-message',
            '#comments', '.comments-area', 'section.comments'
        ]
        found_comments = []
        for selector in comment_selectors:
            elements = soup.select(selector)
            for element in elements:
                is_nested_comment_container = False
                for inner_selector in comment_selectors:
                    if selector != inner_selector and element.select_one(inner_selector):
                        is_nested_comment_container = True
                        break
                if not is_nested_comment_container:
                    comment_text = element.get_text(separator=' ', strip=True)
                    if comment_text and len(comment_text) > 10:
                        found_comments.append(comment_text)
        if found_comments:
            print(f"找到 {len(found_comments)} 条可能的评论内容:")
            for i, comment in enumerate(set(found_comments)):
                print(f"--- 评论 {i+1} ---")
                print(comment)
                print("--------------------")
        else:
            print("未能自动提取到明确的评论内容。")
            print("提示: 评论可能是动态加载的 (需要JavaScript执行)，或者使用了网站特定的HTML结构。")
            print("VIP内容或需要登录的评论也可能无法直接获取。")
    except requests.exceptions.RequestException as e:
        print(f"请求页面内容失败: {e}")
    except Exception as e:
        print(f"处理评论时发生错误: {e}")

def crawl_videos(target_url, headers_dict, html_content):
    print(f"正在尝试从 {target_url} 爬取视频到本地...")
    soup = BeautifulSoup(html_content, 'html.parser')
    found_video_sources = set()
    processed_page_url_for_video = target_url
    if processed_page_url_for_video.startswith("http://"):
        processed_page_url_for_video = "https://" + processed_page_url_for_video[len("http://"):]
    elif not processed_page_url_for_video.startswith("https://") and "://" not in processed_page_url_for_video:
        processed_page_url_for_video = "https://" + processed_page_url_for_video
    base_domain_url_video = ""
    match_domain_video = re.match(r'(https?://[^/]+)', processed_page_url_for_video)
    if match_domain_video:
        base_domain_url_video = match_domain_video.group(1)
        if base_domain_url_video.startswith("http://"):
            base_domain_url_video = "https://" + base_domain_url_video[len("http://"):]
    else:
        domain_part = processed_page_url_for_video.split('/')[0]
        if "://" not in domain_part:
            base_domain_url_video = "https://" + domain_part
        else:
            print(f"警告: 无法从 {processed_page_url_for_video} 解析域名。根相对路径的图片可能无法正确处理。")
    current_path_base_video = processed_page_url_for_video
    if not current_path_base_video.endswith('/'):
        last_slash_idx = current_path_base_video.rfind('/')
        if last_slash_idx > current_path_base_video.find("://") + 2:
            current_path_base_video = current_path_base_video[:last_slash_idx + 1]
        else:
            current_path_base_video += '/'
    if current_path_base_video.startswith("http://"):
        current_path_base_video = "https://" + current_path_base_video[len("http://"):]
    elif not current_path_base_video.startswith("https://") and "://" not in current_path_base_video.split('/')[0]:
        current_path_base_video = "https://" + current_path_base_video
    def resolve_video_url(img_url_param):
        final_url_res = ""
        if img_url_param.startswith("data:"):
            return None
        elif img_url_param.startswith("//"):
            final_url_res = f"https:{img_url_param}"
        elif img_url_param.startswith("/"):
            if base_domain_url_video:
                final_url_res = f"{base_domain_url_video}{img_url_param}"
            else:
                final_url_res = img_url_param
        elif img_url_param.startswith("http://"):
            final_url_res = "https://" + img_url_param[len("http://"):]
        elif img_url_param.startswith("https://"):
            final_url_res = img_url_param
        else:
            if current_path_base_video:
                final_url_res = current_path_base_video + img_url_param
                if final_url_res.startswith("http://"):
                     final_url_res = "https://" + final_url_res[len("http://"):]
            else:
                final_url_res = img_url_param
        if final_url_res:
            if final_url_res.startswith("http://"):
                final_url_res = "https://" + final_url_res[len("http://"):]
            elif not final_url_res.startswith("https://") and not final_url_res.startswith("data:") and "://" not in final_url_res.split('/')[0]:
                 final_url_res = "https://" + final_url_res
            return final_url_res
        return None
    for video_tag in soup.find_all('video'):
        src = video_tag.get('src')
        if src:
            resolved = resolve_video_url(src)
            if resolved: found_video_sources.add(resolved)
        poster = video_tag.get('poster')
        if poster:
            resolved = resolve_video_url(poster)
            if resolved: found_video_sources.add(resolved)
        for source_tag in video_tag.find_all('source'):
            src = source_tag.get('src')
            if src:
                resolved = resolve_video_url(src)
                if resolved: found_video_sources.add(resolved)
    for iframe_tag in soup.find_all(['iframe', 'embed']):
        src = iframe_tag.get('src')
        if src:
            resolved = resolve_video_url(src)
            if resolved: found_video_sources.add(resolved)
    generic_video_pattern = r"['\"`][^'\"`]*?\.(?:mp4|webm|ogg|mov|avi|flv|wmv|mkv|3gp)['\"`]"
    js_video_pattern = r"video[Uu]rl\s*[:=]\s*['\"`][^'\"`]+?['\"`]"
    generic_src_pattern = r"src\s*[:=]\s*['\"`][^'\"`]+?['\"`]"
    for pattern in [generic_video_pattern, js_video_pattern, generic_src_pattern]:
        for match in re.finditer(pattern, html_content, re.IGNORECASE):
            url_match = match.group(1)
            if url_match and not url_match.startswith("data:"):
                resolved = resolve_video_url(url_match)
                if resolved:
                    if any(ext in resolved.lower() for ext in ['.js', '.css', '.html', '.php', '.json']):
                        continue
                    if any(ext in resolved.lower() for ext in ['.mp4', '.webm', '.ogg', '.mov', '.avi', '.flv', '.wmv', '.mkv', '.3gp', '.m3u8']) or \
                       ('video' in resolved.lower() or 'stream' in resolved.lower() or 'embed' in resolved.lower() or 'player' in resolved.lower()) and not '.js' in resolved.lower():
                        found_video_sources.add(resolved)
    print("正在尝试检测可能的m3u8播放列表...")
    m3u8_pattern = r"['\"`][^'\"`]*?\.m3u8[^'\"`]*?['\"`]"
    for match in re.finditer(m3u8_pattern, html_content, re.IGNORECASE):
        m3u8_url = match.group(1)
        resolved_m3u8 = resolve_video_url(m3u8_url)
        if resolved_m3u8:
            print(f"发现可能的m3u8播放列表 (已处理): {resolved_m3u8}")
            found_video_sources.add(resolved_m3u8)
    all_urls = list(found_video_sources)
    if not all_urls:
        print("未找到直接可见的视频链接或m3u8播放列表。")
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
                if len(extension) > 5 or extension == '.html' or extension == '.php' or extension == '.js' or extension == '.css' or extension == '.json':
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
                    response = requests.get(processed_url, headers=download_headers, stream=True, timeout=DEFAULT_TIMEOUT)
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type.lower() or 'application/javascript' in content_type.lower() or 'text/javascript' in content_type.lower() or 'text/css' in content_type.lower():
                        print(f"警告: 返回的是{content_type}内容而非视频文件")
                        break
                    if not ('video' in content_type.lower() or 'stream' in content_type.lower() or 'application/' in content_type.lower()):
                        print(f"警告: 内容类型 {content_type} 可能不是视频")
                    total_size = int(response.headers.get('Content-Length', 0))
                    if total_size < 10000:
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
                        time.sleep(2)
                    else:
                        print("下载失败，已达到最大重试次数")
        except Exception as e:
            print(f"处理视频URL时出错: {e}")
    print(f"\n视频爬取任务完成。下载的文件保存在 {os.path.abspath(download_dir)} 目录中")
    print("注意: 某些下载内容可能不完整或需要进一步处理才能播放")

def crawl_document(target_url, headers_dict):
    print(f"\n=== 开始文档爬取: {target_url} ===")
    download_dir = "downloaded_documents"
    os.makedirs(download_dir, exist_ok=True)
    CONTENT_TYPE_MAP = {
        'application/pdf': '.pdf',
        'text/plain': '.txt',
        'text/html': '.html',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    }
    retry_count = 3
    for attempt in range(1, retry_count+1):
        try:
            print(f"\n尝试下载 ({attempt}/{retry_count})...")
            response = requests.get(target_url, headers=headers_dict,
                                  timeout=DEFAULT_TIMEOUT, stream=True, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
            content_length = int(response.headers.get('Content-Length', 0))
            file_name = f"document_{int(time.time())}"
            if content_type in CONTENT_TYPE_MAP:
                file_ext = CONTENT_TYPE_MAP[content_type]
            else:
                path = urlparse(target_url).path
                file_ext = os.path.splitext(path)[1][:5]
                file_ext = file_ext if file_ext else '.dat'
            file_name += file_ext
            full_path = os.path.join(download_dir, file_name)
            print(f"内容类型: {content_type}")
            print(f"文件大小: {content_length/1024:.1f} KB" if content_length else "文件大小: 未知")
            print(f"保存路径: {os.path.abspath(full_path)}")
            downloaded_size = 0
            start_time = time.time()
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if time.time() - start_time > 1:
                            speed = downloaded_size / (time.time() - start_time) / 1024
                            print(f"\r下载进度: {downloaded_size/1024:.1f} KB ({speed:.1f} KB/s)", end='')
                            start_time = time.time()
            final_size = os.path.getsize(full_path)
            if content_length > 0 and final_size != content_length:
                print(f"\n警告: 文件大小不匹配 (预期: {content_length} 字节, 实际: {final_size} 字节)")
            if final_size < 1024:
                print("警告: 文件尺寸过小，可能是错误页面")
            print(f"\n文档下载成功: {file_name}")
            return
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误 ({e.response.status_code}): {str(e)}")
            if attempt == retry_count:
                print("已达到最大重试次数")
                return
        except requests.exceptions.Timeout:
            print("请求超时")
            if attempt == retry_count:
                print("无法在超时时间内完成下载")
                return
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {str(e)}")
            return
        except Exception as e:
            print(f"未知错误: {str(e)}")
            return
        if attempt < retry_count:
            print("等待2秒后重试...")
            time.sleep(2)

def load_counter():
    try:
        with open('.request_counter', 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def save_counter(count):
    with open('.request_counter', 'w') as f:
        f.write(str(count))

def crawl_douban_top250_movies(headers_dict):
    print("\n--- 开始爬取豆瓣电影Top250 ---")
    try:
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True)
        column_headers = ['名称', '图片', '排名', '评分', '信息', '简介']
        for col, header_text in enumerate(column_headers):
            sheet.write(0, col, header_text)
        current_row = 1
        def request_douban_page(url, page_num_for_print, headers):
            print(f"正在请求豆瓣Top250第 {page_num_for_print}/10 页: {url}")
            try:
                response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                print(f"请求豆瓣页面失败: {url}, 错误: {e}")
                return None
        def parse_and_save_page(soup, sheet_obj, current_row_num_ref):
            movie_list = soup.find(class_='grid_view').find_all('li')
            if not movie_list:
                print("未能找到电影列表 (class 'grid_view')，页面结构可能已更改或当前页面无内容。")
                return current_row_num_ref
            for item in movie_list:
                item_name_tag = item.find(class_='title')
                item_name = item_name_tag.string if item_name_tag else "N/A"
                item_img_tag = item.find('a')
                item_img = item_img_tag.find('img').get('src') if item_img_tag and item_img_tag.find('img') else "N/A"
                index_tag = item.find('em', class_='')
                item_index = index_tag.string if index_tag else "N/A"
                score_tag = item.find(class_='rating_num')
                item_score = score_tag.string if score_tag else "N/A"
                p_tag = item.find('p', class_='')
                item_author_info = ""
                if p_tag:
                    item_author_info = ' '.join(p_tag.text.split())
                item_intr_tag = item.find(class_='inq')
                item_intr = item_intr_tag.string if item_intr_tag else 'NOT AVAILABLE'
                print(f"  爬取电影：{item_index} | {item_name} | {item_score}")
                data_to_write = [item_name, item_img, item_index, item_score, item_author_info, item_intr]
                for col, data_val in enumerate(data_to_write):
                    sheet_obj.write(current_row_num_ref, col, data_val)
                current_row_num_ref += 1
            return current_row_num_ref
        for i in range(0, 10):
            page_num_for_display = i + 1
            url = f'https://movie.douban.com/top250?start={i * 25}&filter='
            douban_headers = {'User-Agent': headers_dict.get('User-Agent', 'Mozilla/5.0')}
            html_content = request_douban_page(url, page_num_for_display, douban_headers)
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                current_row = parse_and_save_page(soup, sheet, current_row)
            else:
                print(f"跳过豆瓣Top250第 {page_num_for_display} 页，因为未能获取内容。")
            if i < 9:
                 sleep_duration = random.uniform(1, 3)
                 print(f"暂停 {sleep_duration:.2f} 秒...")
                 time.sleep(sleep_duration)
        save_dir = "downloaded_douban"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '豆瓣电影Top250.xlsx')
        book.save(save_path)
        print(f"\n豆瓣电影Top250数据已成功保存到: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"爬取豆瓣电影Top250过程中发生错误: {e}")
        traceback.print_exc()
    print("--- 豆瓣电影Top250爬取任务结束 ---")

def main():
    request_counter = load_counter()
    if request_counter >= 5:
        print("\n已达到5次使用（或更多），需要更新请求头以继续。")
        custom_headers = {'User-Agent': input("请重新输入有效的User-Agent (例如: Mozilla/5.0 ...): ").strip()}
        if not custom_headers['User-Agent']:
            print("错误: User-Agent是必填项。程序将退出。")
            return
        save_counter(0)
        request_counter = 0
        print("User-Agent已更新，计数器已重置。")
    else:
        custom_headers = {'User-Agent': input("请输入User-Agent (例如: Mozilla/5.0 ...): ").strip()}
        if not custom_headers['User-Agent']:
            print("错误: User-Agent是必填项。程序将退出。")
            return
    print("\n请选择操作:")
    print("1. 爬取图片链接")
    print("2. 爬取评论")
    print("3. 爬取视频到本地")
    print("4. 爬取豆瓣电影Top250信息")
    print("5. 爬取文档内容")
    choice = input("请输入选项 (1-5): ").strip()
    processed_target_url = ""
    html_content_main = None
    if choice == '4':
        crawl_douban_top250_movies(custom_headers)
        request_counter += 1
        save_counter(request_counter)
        print("\n本次豆瓣爬取任务使用已记录。")
        print("--- 任务结束 ---")
        return
    if choice in ['1', '2', '3', '5']:
        target_url_input = input("\n请输入目标网址: ").strip()
        if not target_url_input:
            print("错误: 未输入目标网址。程序将退出。")
            return
        processed_target_url = target_url_input
        if not processed_target_url.startswith("http://") and not processed_target_url.startswith("https://"):
            if "://" in processed_target_url:
                print(f"错误: 不支持的URL协议。请输入以 http:// 或 https:// 开头的网址，或纯域名。")
                return
            processed_target_url = "https://" + processed_target_url
        elif processed_target_url.startswith("http://"):
            processed_target_url = "https://" + processed_target_url[len("http://"):]
        print(f"目标URL (将使用HTTPS协议): {processed_target_url}")
        if choice == '3' and 'bilibili.com' in processed_target_url:
            print("\n检测到B站链接，尝试专门解析视频...")
            match = re.search(r'bvid=([^&]+)', processed_target_url) or re.search(r'/video/(BV\w+)', processed_target_url)
            if match:
                bvid = match.group(1)
                parse_bilibili_video(bvid, processed_target_url, custom_headers)
            else:
                print("无法从URL中提取B站视频ID。")
            request_counter += 1
            save_counter(request_counter)
            print("\n本次B站视频解析任务使用已记录。")
            print("--- 任务结束 ---")
            return
        if choice in ['1', '2', '3']:
            try:
                print(f"\n正在请求目标页面: {processed_target_url} ...")
                response = requests.get(processed_target_url, headers=custom_headers, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                response.encoding = response.apparent_encoding
                html_content_main = response.text
                print(f"成功获取目标页面内容。")
            except requests.exceptions.Timeout:
                print(f"请求超时: 无法在{DEFAULT_TIMEOUT}秒内连接到 {processed_target_url}")
                return
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP错误: {http_err} - {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
                return
            except requests.exceptions.RequestException as e:
                print(f"请求目标页面失败: {e}")
                return
    if choice == '1':
        print("\n--- 开始爬取图片链接 ---")
        if html_content_main:
            crawl_images_revised(processed_target_url, html_content_main, custom_headers)
        else:
            print("未能获取页面内容，无法爬取图片。")
    elif choice == '2':
        print("\n--- 开始爬取评论 ---")
        crawl_comments(processed_target_url, custom_headers)
    elif choice == '3':
        if 'bilibili.com' not in processed_target_url:
            print("\n--- 开始爬取视频到本地 ---")
            if html_content_main:
                crawl_videos(processed_target_url, custom_headers, html_content_main)
            else:
                print("未能获取页面内容，无法爬取视频。")
    elif choice == '5':
        print("\n--- 开始爬取文档内容 ---")
        crawl_document(processed_target_url, custom_headers)
    elif choice not in ['1','2','3','4','5']:
        print("无效的选项。程序将退出。")
        return
    if choice in ['1','2','3','5']:
        request_counter += 1
        save_counter(request_counter)
        print("\n本次任务使用已记录。")
    print("--- 主任务结束 ---")

if __name__ == "__main__":
    main()







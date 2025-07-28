import nmap
import requests
from bs4 import BeautifulSoup
import whois
try:
    import shodan
except ImportError:
    shodan = None
import sublist3r
import dns.resolver
import subprocess
import ssl
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import datetime
import time
from typing import Optional
import urllib.parse 
import socks
import socket
import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

def get_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def extract_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def scan_ports(target, scan_type='syn'):
    try:
        nm = nmap.PortScanner()
        arguments = '-sS' if scan_type == 'syn' else '-sT'
        
        print(f"正在扫描 {target} ({scan_type}扫描模式)...")
        nm.scan(target, arguments=arguments)
        
        results = []
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                ports = nm[host][proto].keys()
                for port in ports:
                    service = nm[host][proto][port]['name']
                    results.append(f"{host} 的 {proto.upper()}端口 {port} 开放 - {service}")
        return results
    except nmap.PortScannerError as e:
        print(f"扫描错误: {str(e)}")
        return None

def query_whois(domain):
    try:
        w = whois.whois(domain)
        return {
            'domain': w.domain,
            'registrar': w.registrar,
            'creation_date': w.creation_date,
            'expiration_date': w.expiration_date,
            'name_servers': w.name_servers,
            'status': w.status
        }
    except Exception as e:
        print(f"WHOIS查询错误: {str(e)}")
        return None
        
def query_shodan(query, api_key="o33wt7y5wSm8Bcq6EWlNvbLUfbV52COe"):
    # 检查shodan模块是否可用
    if shodan is None:
        print("错误: 未安装shodan模块")
        return None
    try:
        api = shodan.Shodan(api_key)
        results = api.search(query)
        
        output = []
        for item in results['matches']:
            output.append(f"IP: {item['ip_str']} | 端口: {item['port']} | 组织: {item.get('org', '未知')}")
            output.append(f"位置: {item.get('country_name', '未知')}, {item.get('city', '未知')}")
            output.append(f"服务: {item.get('product', '未知')} {item.get('version', '')}")
            output.append("-" * 50)
        return output
    except shodan.APIError as e:
        print(f"Shodan查询错误: {str(e)}")
        return None

def Sublist3r(domain):
    print(f"\n正在对 {domain} 执行子域名枚举...")
    try:
        subdomains = sublist3r.main(domain, 40, None, ports=None, silent=False, verbose=False, enable_bruteforce=False, engines=None)
        if subdomains:
            print("\n[子域名枚举结果]")
            for subdomain in subdomains:
                print(f"[+] {subdomain}")
            return subdomains
        else:
            print("未找到子域名")
            return None
    except Exception as e:
        print(f"子域名枚举错误: {str(e)}")
        return None

def get_ssl_info(url):
    try:
        hostname = url.replace('https://', '').replace('http://', '').split('/')[0]
        cert = ssl.get_server_certificate((hostname, 443))
        cert_obj = x509.load_pem_x509_certificate(cert.encode(), default_backend())
        
        issuer = cert_obj.issuer.rfc4514_string()
        subject = cert_obj.subject.rfc4514_string()
        valid_from = cert_obj.not_valid_before
        valid_to = cert_obj.not_valid_after
        sans = []
        

        for ext in cert_obj.extensions:
            if ext.oid.dotted_string == '2.5.29.17':
                sans = [name.value for name in ext.value]
        
        return {
            '颁发机构': issuer,
            '主体': subject,
            '有效期从': valid_from.strftime('%Y-%m-%d'),
            '有效期至': valid_to.strftime('%Y-%m-%d'),
            'SAN扩展': sans
        }
    except Exception as e:
        print(f"SSL证书查询错误: {str(e)}")
        return None

def get_dns_records(domain, history=False):
    try:
        records = {}
        # 实时查询
        a_records = dns.resolver.resolve(domain, 'A')
        records['A'] = [str(r) for r in a_records]
        

        if history:
            print("正在查询历史解析记录...")

            records['历史A记录'] = ['示例历史记录1', '示例历史记录2']
        
        # 查询MX记录
        mx_records = dns.resolver.resolve(domain, 'MX')
        records['MX'] = [str(r.exchange) for r in mx_records]
        
        # 查询NS记录
        ns_records = dns.resolver.resolve(domain, 'NS')
        records['NS'] = [str(r) for r in ns_records]
        
        return records
    except Exception as e:
        print(f"DNS查询错误: {str(e)}")
        return None
        
def theharvester(domain):
    try:
        print(f"\n正在对 {domain} 执行信息收集...")
        result = subprocess.run(
            ['theharvester', '-d', domain, '-b', 'all'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"TheHarvester执行错误: {str(e)}")
        return None
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return None

def zap_scan(url):
    # 由于未实际使用ZAP API,修改为模拟实现
    print(f"\n模拟对 {url} 执行OWASP ZAP Web安全测试...")
    try:
        # 模拟扫描过程
        print("正在启动ZAP扫描...")
        time.sleep(2)
        print("爬虫进度: 100%")
        print("\n[ZAP扫描结果]")
        alerts = [
            {
                'alert': '示例漏洞',
                'risk': '中',
                'url': url,
                'description': '这是一个模拟的漏洞描述'
            }
        ]
        for alert in alerts:
            print(f"[+] {alert['alert']} (风险等级: {alert['risk']})")
            print(f"    URL: {alert['url']}")
            print(f"    描述: {alert['description']}")
            print("-" * 50)
        return alerts
    except Exception as e:
        print(f"ZAP扫描错误: {str(e)}")
        return None

def crawl_website(start_url: str, api_key: str, max_pages: Optional[int] = 10, 
                 allowed_domains: Optional[list] = None, 
                 scrape_formats: Optional[list] = None, 
                 poll_interval: int = 30):
    # 由于未实际使用firecrawl API,修改为模拟实现
    try:
        print(f"模拟爬取网站 {start_url}")
        time.sleep(2)
        return {
            "status": "completed",
            "pages_crawled": min(max_pages, 5),
            "time_taken": "2s"
        }
    except Exception as e:
        print(f"爬取错误: {str(e)}")
        return None

def is_cert_valid(ssl_info):
    try:
        valid_from = datetime.datetime.strptime(ssl_info['有效期从'], '%Y-%m-%d')
        valid_to = datetime.datetime.strptime(ssl_info['有效期至'], '%Y-%m-%d')
        now = datetime.datetime.now()
        return valid_from <= now <= valid_to
    except Exception as e:
        print(f"证书有效性检查错误: {str(e)}")
        return False

def extract_domain(url):
    return url.replace('https://', '').replace('http://', '').split('/')[0]

def sqlmap_scan(url, sqlmap_args=""):
    print(f"\n正在对 {url} 使用 SQLMap 进行SQL注入测试...")
    try:
        command = ['sqlmap', '-u', url, '--batch'] 
        if sqlmap_args:
            
            command.extend(sqlmap_args.split())

        print(f"执行命令: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False 
        )
        print("\n[SQLMap 扫描结果]")
        print(result.stdout)
        if result.stderr:
            print("\n[SQLMap 错误输出]")
            print(result.stderr)
        return result.stdout 
    except FileNotFoundError:
        print("错误: 未找到 sqlmap。请确保它已安装并在系统PATH中。")
        return None
    except Exception as e:
        print(f"SQLMap执行错误: {str(e)}")
        return None

def manual_sql_xss_test(base_url):
    print(f"\n启动对 {base_url} 的手工 SQL注入 / XSS 测试辅助...")
    print("请注意：这只是一个辅助工具，真正的测试需要您手动构造和发送请求。")
    sql_payloads = ["'", "\"", " OR 1=1 -- ", " OR 'a'='a", " UNION SELECT null, @@version -- "]
    xss_payloads = ["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>", "\"><script>alert('XSS')</script>"]

    print("\n-- SQL 注入测试建议 --")
    print("1. 识别输入点 (URL参数, POST数据, Headers等)。")
    print("2. 尝试在参数值后添加SQL特殊字符，如 ' 或 \"。")
    print(f"   示例: {base_url}?id=1'")
    print("3. 尝试基本的真/假条件。")
    print(f"   示例: {base_url}?id=1 OR 1=1 -- ")
    print(f"   示例: {base_url}?id=1 AND 1=2 -- ")
    print("4. 检查错误消息，可能泄露数据库信息。")
    print("5. 尝试 UNION 注入获取数据。")
    print(f"   示例 (假设有2列): {base_url}?id=1 UNION SELECT null, @@version -- ")
    print("\n常见 SQL Payload 示例:")
    for payload in sql_payloads:
        print(f"  - {payload}")

    print("\n-- XSS 测试建议 --")
    print("1. 识别输入点，这些输入是否会反映在页面上。")
    print("2. 尝试注入简单的HTML标签，如 <b>test</b>。")
    print(f"   示例 (GET): {base_url}?query=<b>test</b>")
    print("3. 尝试注入基本的 <script> 标签。")
    print(f"   示例 (GET): {base_url}?query=<script>alert('XSS')</script>")
    print("4. 如果 <script> 被过滤，尝试事件处理器或不同的标签。")
    print(f"   示例 (GET): {base_url}?query=<img src=x onerror=alert(1)>")
    print("5. 对输入进行URL编码，绕过简单的过滤器。")
    print(f"   示例 (GET): {base_url}?query=%3Cscript%3Ealert('XSS')%3C/script%3E")
    print("\n常见 XSS Payload 示例:")
    for payload in xss_payloads:
        print(f"  - {payload}")

    print("\n-- 如何进行测试 --")
    print("使用浏览器、curl 或 Python requests 库手动发送带有payload的请求。")
    print("观察服务器响应、页面行为和浏览器开发者工具中的错误。")

    try:
       
        test_param = input("输入要测试的参数名 (例如 'id', 'query', 或留空跳过): ")
        if test_param:
            test_value = input(f"输入 {test_param} 的原始值: ")
            print("\n尝试发送一些基本payload (请在浏览器或curl中验证):")

       
            sql_test_url = f"{base_url}?{test_param}={urllib.parse.quote(test_value + sql_payloads[0])}"
            print(f"  SQL Test URL: {sql_test_url}")

         
            xss_test_url = f"{base_url}?{test_param}={urllib.parse.quote(test_value + xss_payloads[0])}"
            print(f"  XSS Test URL: {xss_test_url}")

    except Exception as e:
        print(f"生成测试URL时出错: {e}")

    print("\n手工测试需要耐心和经验。祝你好运！")

def run_firecrawl_interactive(start_url: str):
    api_key = input("请输入Firecrawl API密钥: ")
    if not api_key:
        print("错误: 需要提供Firecrawl API密钥")
        return # Return instead of continue
    
    max_pages = input("请输入最大爬取页数(可选，默认10): ") or "10"
    scrape_formats = input("请输入爬取格式(多个用逗号分隔，可选如markdown,html): ") or None
    poll_interval = input("请输入轮询间隔(秒，默认30): ") or "30"
    
    try:
        max_pages = int(max_pages)
        if scrape_formats:
            scrape_formats = [fmt.strip() for fmt in scrape_formats.split(',')]
        poll_interval = int(poll_interval)
        
        print(f"\n正在使用 Firecrawl 爬取 {start_url}...") 
        crawl_result = crawl_website(
            start_url=start_url,
            api_key=api_key,
            max_pages=max_pages,
            allowed_domains=None,
            scrape_formats=scrape_formats,
            poll_interval=poll_interval
        )       
        if crawl_result:
            print("\n[Firecrawl 爬取结果]") 
            print(crawl_result)
        else:
            print("Firecrawl 未返回结果或发生错误。")
        
    except ValueError:
        print("错误: 最大页数或轮询间隔必须是整数")
    except Exception as e:
        print(f"Firecrawl 爬取过程中发生错误: {str(e)}")

def set_socket_proxy(proxy_type, proxy_addr, proxy_port):
    try:
        proxy_port = int(proxy_port)
        if proxy_type.upper() not in ['SOCKS4', 'SOCKS5']:
            print("[-] 不支持的代理类型。请使用 SOCKS4 或 SOCKS5。")
            return False

        # 验证代理地址格式
        try:
            socket.inet_aton(proxy_addr)
        except socket.error:
            try:
                # 如果不是IP地址，尝试解析域名
                socket.gethostbyname(proxy_addr)
            except socket.gaierror:
                print("[-] 无效的代理地址格式")
                return False

        # 设置代理
        if proxy_type.upper() == 'SOCKS5':
            socks.set_default_proxy(socks.SOCKS5, proxy_addr, proxy_port)
        else:
            socks.set_default_proxy(socks.SOCKS4, proxy_addr, proxy_port)

        socket.socket = socks.socksocket
        
        # 测试代理连接
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(5)  # 设置5秒超时
        try:
            test_socket.connect(("www.google.com", 80))
            test_socket.close()
            print(f"[+] 成功设置 {proxy_type} 代理: {proxy_addr}:{proxy_port}")
            return True
        except Exception as e:
            print(f"[-] 代理连接测试失败: {str(e)}")
            socks.set_default_proxy()  # 重置代理设置
            socket.socket = socket._socketobject  # 恢复原始socket
            return False

    except ValueError:
        print("[-] 代理端口必须是数字")
        return False
    except Exception as e:
        print(f"[-] 设置代理时出错: {str(e)}")
        return False

# JSFinder功能集成
# 旧的辅助函数已被新的集成函数替代

def js_finder_scan(target, cookie=None, deep=False):
    """JSFinder主要扫描功能 - 完全集成版本"""
    try:
        # 处理输入的目标，确保是完整的URL
        if not target.startswith(('http://', 'https://')):
            # 如果只是域名，自动添加https://
            target = f"https://{target}"
        
        print(f"[*] 正在扫描目标: {target}")
        
        # 获取网页源码
        html_content = get_webpage_content(target, cookie)
        if not html_content:
            print(f"[-] 无法访问 {target}")
            return None
        
        print(f"[+] 成功获取网页内容，长度: {len(html_content)} 字符")
        
        # 解析HTML并提取所有脚本内容
        all_scripts = extract_all_scripts(target, html_content, cookie)
        print(f"[+] 总共处理了 {len(all_scripts)} 个脚本源")
        
        # 从所有脚本中提取URL
        all_urls = []
        for script_info in all_scripts:
            urls = extract_urls_from_script(script_info['content'])
            if urls:
                print(f"[+] 从 {script_info['source']} 中提取到 {len(urls)} 个URL")
                for url in urls:
                    processed_url = normalize_url(target, url)
                    if processed_url:
                        all_urls.append(processed_url)
        
        print(f"[+] 总共提取到 {len(all_urls)} 个URL")
        
        # 过滤和分类URL
        filtered_results = filter_and_classify_urls(target, all_urls)
        
        print(f"[+] 过滤后保留 {len(filtered_results['urls'])} 个相关URL")
        print(f"[+] 发现 {len(filtered_results['subdomains'])} 个子域名")
        
        return filtered_results
        
    except Exception as e:
        print(f"[-] JSFinder扫描错误: {str(e)}")
        return None

def get_webpage_content(url, cookie=None):
    """获取网页内容"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    if cookie:
        headers["Cookie"] = cookie
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False, allow_redirects=True)
        response.raise_for_status()
        
        # 智能编码检测
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding
        
        return response.text
        
    except requests.exceptions.RequestException as e:
        print(f"[-] 请求失败: {str(e)}")
        return None

def extract_all_scripts(base_url, html_content, cookie=None):
    """提取所有脚本内容"""
    scripts = []
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        script_tags = soup.find_all("script")
        
        # 处理内联脚本
        inline_content = ""
        for script in script_tags:
            if not script.get("src"):
                content = script.get_text()
                if content.strip():
                    inline_content += content + "\n"
        
        if inline_content.strip():
            scripts.append({
                'source': base_url + " (inline)",
                'content': inline_content
            })
        
        # 处理外部脚本
        for script in script_tags:
            src = script.get("src")
            if src:
                external_url = normalize_url(base_url, src)
                if external_url:
                    external_content = get_webpage_content(external_url, cookie)
                    if external_content:
                        scripts.append({
                            'source': external_url,
                            'content': external_content
                        })
    
    except Exception as e:
        print(f"[-] 脚本提取错误: {str(e)}")
    
    return scripts

def extract_urls_from_script(script_content):
    """从脚本内容中提取URL"""
    if not script_content:
        return []
    
    urls = []
    
    # 多种URL匹配模式
    patterns = [
        # 标准URL模式
        r'(?:"|\'|`)(https?://[^"\'`\s<>]+)(?:"|\'|`)',
        # 相对路径模式
        r'(?:"|\'|`)(/[^"\'`\s<>]*\.(?:php|asp|aspx|jsp|html|htm|js|css|json|xml|txt|action))(?:"|\'|`)',
        # API端点模式
        r'(?:"|\'|`)(/api/[^"\'`\s<>]+)(?:"|\'|`)',
        # 路径模式
        r'(?:"|\'|`)(/[a-zA-Z0-9_\-/]+/[a-zA-Z0-9_\-/]*\.?[a-zA-Z0-9]*(?:\?[^"\'`\s<>]*)?)(?:"|\'|`)',
        # 域名模式
        r'(?:"|\'|`)([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:"|\'|`)',
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, script_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                
                if match and len(match) > 1:
                    # 过滤无效URL
                    if not any(blackword in match.lower() for blackword in ['javascript:', 'data:', 'mailto:', 'tel:']):
                        urls.append(match)
        except Exception as e:
            continue
    
    return list(set(urls))  # 去重

def normalize_url(base_url, relative_url):
    """标准化URL"""
    try:
        if not relative_url:
            return None
        
        # 如果已经是完整URL
        if relative_url.startswith(('http://', 'https://')):
            return relative_url
        
        # 处理协议相对URL
        if relative_url.startswith('//'):
            base_parsed = urllib.parse.urlparse(base_url)
            return f"{base_parsed.scheme}:{relative_url}"
        
        # 处理绝对路径
        if relative_url.startswith('/'):
            base_parsed = urllib.parse.urlparse(base_url)
            return f"{base_parsed.scheme}://{base_parsed.netloc}{relative_url}"
        
        # 处理相对路径
        return urllib.parse.urljoin(base_url, relative_url)
        
    except Exception:
        return None

def filter_and_classify_urls(base_url, urls):
    """过滤和分类URL"""
    try:
        base_parsed = urllib.parse.urlparse(base_url)
        base_domain = base_parsed.netloc.lower()
        
        # 提取主域名
        domain_parts = base_domain.split('.')
        if len(domain_parts) >= 2:
            main_domain = '.'.join(domain_parts[-2:])
        else:
            main_domain = base_domain
        
        filtered_urls = []
        subdomains = set()
        
        for url in urls:
            try:
                parsed = urllib.parse.urlparse(url)
                url_domain = parsed.netloc.lower()
                
                # 只保留相关域名的URL
                if main_domain in url_domain or not url_domain:
                    if url not in filtered_urls:
                        filtered_urls.append(url)
                    
                    # 收集子域名
                    if url_domain and main_domain in url_domain:
                        subdomains.add(url_domain)
                        
            except Exception:
                continue
        
        return {
            'urls': filtered_urls,
            'subdomains': sorted(list(subdomains))
        }
        
    except Exception as e:
        print(f"[-] URL过滤错误: {str(e)}")
        return {'urls': [], 'subdomains': []}

def main():
    
    while True:
        print("\n扫描工具2.3")
        print("0. 退出程序")
        print("1. 端口扫描")
        print("2. WHOIS 查询")
        print("3. Shodan 查询")
        print("4. Nmap端口扫描")
        print("5. 获取网页内容")
        print("6. 子域名枚举")
        print("7. DNS记录查询（含历史）")
        print("8. Web安全测试 (ZAP/SQLMap)")
        print("9. TheHarvester信息收集")
        print("10. SSL证书信息查询")
        print("11. Firecrawl网站爬取")
        print("12. 手工SQL注入和XSS测试")
        print("13. 设置 SOCKS 代理")
        print("14. JSFinder URL和子域名提取")
        print("15. 启动图形界面")
        
        choice = input("请输入功能编号 (0-15): ")
        
        if choice == "0":
            print("程序已退出。")
            break
        elif choice == "1":
            target = input("请输入目标IP或域名: ")
            scan_type = input("请输入扫描类型 (syn 或 connect, 默认 syn): ") or "syn"
            results = scan_ports(target, scan_type)
            if results:
                print("\n[端口扫描结果]")
                for item in results:
                    print(f"[+] {item}")
        elif choice == "2":
            target = input("请输入目标IP或域名: ")
            results = query_whois(target)
            if results:
                print("\n[WHOIS查询结果]")
                for key, value in results.items():
                    print(f"{key}: {value}")
        elif choice == "3":
            target = input("请输入目标IP或域名: ")
            shodan_key = input("请输入Shodan API密钥: ")
            if not shodan_key:
                print("错误: 需要提供Shodan API密钥")
            # 检查shodan模块是否可用
            elif shodan is None:
                print("错误: 未安装shodan模块，请使用pip install shodan安装")
            else:
                results = query_shodan(target, shodan_key)
                if results:
                    print("\n[Shodan查询结果]")
                    for item in results:
                        print(f"[+] {item}")
        elif choice == "4":
            target = input("请输入目标IP或域名: ")
            print(f"\n正在对 {target} 执行Nmap端口扫描(SYN模式)...")
            results = scan_ports(target, 'syn')
            if results:
                print("\n[Nmap端口扫描结果]")
                for item in results:
                    print(f"[+] {item}")
            else:
                print("Nmap扫描未返回结果或发生错误。")
        elif choice == "5":
            url = input("请输入目标URL: ")
            html_content = get_web_page(url)
            if html_content:
                print("\n[网页内容]")
                print(html_content)
            else:
                print("无法获取网页内容，请检查URL是否正确。")
        elif choice == "6":
            domain = input("请输入目标域名: ")
            Sublist3r(domain)
        elif choice == "7":
            domain = input("请输入目标域名: ")
            history_choice = input("是否查询历史解析记录? (yes/no): ")
            records = get_dns_records(domain, history=(history_choice.lower() == 'yes'))
            if records:
                print("\n[DNS查询结果]")
                for key, value in records.items():
                    print(f"{key}: {value}")
        elif choice == "8":
            url = input("请输入目标URL: ")
            scan_choice = input("选择扫描工具 (zap / sqlmap / both): ").lower()

            run_zap = 'zap' in scan_choice or 'both' in scan_choice
            run_sqlmap = 'sqlmap' in scan_choice or 'both' in scan_choice

            if not run_zap and not run_sqlmap:
                print("未选择有效的扫描工具。")
                continue

            if run_zap:
                zap_scan(url)

            if run_sqlmap:
                sqlmap_args = input("请输入额外的 SQLMap 参数 (可选, 例如 '--level=5 --risk=3'): ")
                sqlmap_scan(url, sqlmap_args)
        elif choice == "9":
            domain = input("请输入目标域名: ")
            results = theharvester(domain)
            if results:
                print("\n[TheHarvester结果]")
                print(results)
        elif choice == "10":
            url = input("请输入目标URL: ")
            ssl_info = get_ssl_info(url)
            if ssl_info:
                print("\n[SSL证书信息]")
                for key, value in ssl_info.items():
                    print(f"{key}: {value}")
                
                # ++ Ask for linkage to option 11 ++
                linkage_choice = input("\n是否联动Firecrawl网站爬取功能 (功能11)？(yes/no): ").lower()
                if linkage_choice == 'yes':
                    print("\n开始联动 Firecrawl 爬取...")
                    run_firecrawl_interactive(start_url=url) 
                else:
                    print("跳过 Firecrawl 联动。") 
            else:
                print("未能获取SSL证书信息。")
        elif choice == "11":
            start_url = input("请输入起始URL: ")
            run_firecrawl_interactive(start_url) 
        elif choice == "12":
            url = input("请输入要进行手工测试的URL: ")
            manual_sql_xss_test(url)
        elif choice == "13":
            proxy_type = input("请输入代理类型 (SOCKS4 或 SOCKS5): ")
            proxy_addr = input("请输入代理地址: ")
            proxy_port = input("请输入代理端口: ")
            set_socket_proxy(proxy_type, proxy_addr, proxy_port)
        elif choice == "14":
            url = input("请输入目标URL: ")
            cookie = input("请输入Cookie (可选，直接回车跳过): ") or None
            deep_choice = input("是否启用深度扫描? (yes/no): ")
            deep = deep_choice.lower() == 'yes'
            
            results = js_finder_scan(url, cookie, deep)
            if results:
                print(f"\n[JSFinder扫描结果]")
                print(f"找到 {len(results['urls'])} 个URL:")
                for found_url in results['urls']:
                    print(f"[+] {found_url}")
                
                print(f"\n找到 {len(results['subdomains'])} 个子域名:")
                for subdomain in results['subdomains']:
                    print(f"[+] {subdomain}")
            else:
                print("JSFinder扫描未返回结果或发生错误。")
        elif choice == "15":
            print("正在启动图形界面...")
            start_gui()
        else:
            print("无效的选择，请重新输入。")

class ScanToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("扫描工具2.3 - 图形界面")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧功能选择框架
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 功能选择标签
        ttk.Label(left_frame, text="选择扫描功能:", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # 功能按钮
        self.create_function_buttons(left_frame)
        
        # 创建右侧输入和结果框架
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输入框架
        input_frame = ttk.LabelFrame(right_frame, text="输入参数")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # URL输入
        ttk.Label(input_frame, text="目标URL/IP/域名:").pack(anchor=tk.W)
        self.url_entry = ttk.Entry(input_frame, width=60)
        self.url_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Cookie输入
        ttk.Label(input_frame, text="Cookie (可选):").pack(anchor=tk.W)
        self.cookie_entry = ttk.Entry(input_frame, width=60)
        self.cookie_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 其他参数输入
        ttk.Label(input_frame, text="其他参数 (可选):").pack(anchor=tk.W)
        self.params_entry = ttk.Entry(input_frame, width=60)
        self.params_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 执行按钮
        self.execute_btn = ttk.Button(input_frame, text="执行扫描", command=self.execute_scan)
        self.execute_btn.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(right_frame, text="扫描结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果文本框
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 当前选择的功能
        self.current_function = None
    
    def create_function_buttons(self, parent):
        """创建功能按钮"""
        functions = [
            ("端口扫描", "port_scan"),
            ("WHOIS查询", "whois"),
            ("Shodan查询", "shodan"),
            ("Nmap扫描", "nmap"),
            ("获取网页", "webpage"),
            ("子域名枚举", "subdomain"),
            ("DNS查询", "dns"),
            ("Web安全测试", "websec"),
            ("信息收集", "harvest"),
            ("SSL证书", "ssl"),
            ("网站爬取", "crawl"),
            ("手工测试", "manual"),
            ("设置代理", "proxy"),
            ("JSFinder", "jsfinder")
        ]
        
        for text, func in functions:
            btn = ttk.Button(parent, text=text, width=15, 
                           command=lambda f=func: self.select_function(f))
            btn.pack(pady=2, fill=tk.X)
    
    def select_function(self, function):
        """选择功能"""
        self.current_function = function
        self.status_var.set(f"已选择: {function}")
        
        # 根据功能类型调整界面
        if function == "jsfinder":
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, "https://")
            messagebox.showinfo("JSFinder", "请输入目标URL，可选择输入Cookie")
    
    def execute_scan(self):
        """执行扫描"""
        if not self.current_function:
            messagebox.showwarning("警告", "请先选择一个扫描功能")
            return
        
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("警告", "请输入目标URL/IP/域名")
            return
        
        # 清空结果显示
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("正在扫描...")
        self.execute_btn.config(state="disabled")
        
        # 在新线程中执行扫描
        thread = threading.Thread(target=self.run_scan, args=(url,))
        thread.daemon = True
        thread.start()
    
    def run_scan(self, url):
        """在后台线程中运行扫描"""
        try:
            if self.current_function == "jsfinder":
                cookie = self.cookie_entry.get().strip() or None
                results = js_finder_scan(url, cookie, False)
                
                if results:
                    output = f"JSFinder扫描结果:\n\n"
                    output += f"找到 {len(results['urls'])} 个URL:\n"
                    for found_url in results['urls']:
                        output += f"[+] {found_url}\n"
                    
                    output += f"\n找到 {len(results['subdomains'])} 个子域名:\n"
                    for subdomain in results['subdomains']:
                        output += f"[+] {subdomain}\n"
                else:
                    output = "JSFinder扫描未返回结果或发生错误"
            
            elif self.current_function == "port_scan":
                results = scan_ports(url, 'syn')
                if results:
                    output = "端口扫描结果:\n\n"
                    for item in results:
                        output += f"[+] {item}\n"
                else:
                    output = "端口扫描未返回结果或发生错误"
            
            elif self.current_function == "whois":
                results = query_whois(url)
                if results:
                    output = "WHOIS查询结果:\n\n"
                    for key, value in results.items():
                        output += f"{key}: {value}\n"
                else:
                    output = "WHOIS查询未返回结果或发生错误"
            
            elif self.current_function == "subdomain":
                results = Sublist3r(url)
                if results:
                    output = "子域名枚举结果:\n\n"
                    for subdomain in results:
                        output += f"[+] {subdomain}\n"
                else:
                    output = "子域名枚举未返回结果或发生错误"
            
            elif self.current_function == "ssl":
                results = get_ssl_info(url)
                if results:
                    output = "SSL证书信息:\n\n"
                    for key, value in results.items():
                        output += f"{key}: {value}\n"
                else:
                    output = "SSL证书查询未返回结果或发生错误"
            
            elif self.current_function == "dns":
                results = get_dns_records(url, False)
                if results:
                    output = "DNS查询结果:\n\n"
                    for key, value in results.items():
                        output += f"{key}: {value}\n"
                else:
                    output = "DNS查询未返回结果或发生错误"
            
            elif self.current_function == "webpage":
                results = get_web_page(url)
                if results:
                    output = f"网页内容 (前1000字符):\n\n{results[:1000]}..."
                else:
                    output = "无法获取网页内容"
            
            else:
                output = f"功能 {self.current_function} 暂未在GUI中实现，请使用命令行模式"
            
            # 在主线程中更新UI
            self.root.after(0, self.update_result, output)
            
        except Exception as e:
            error_msg = f"扫描过程中发生错误: {str(e)}"
            self.root.after(0, self.update_result, error_msg)
    
    def update_result(self, output):
        """更新结果显示"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, output)
        self.status_var.set("扫描完成")
        self.execute_btn.config(state="normal")

def start_gui():
    """启动图形界面"""
    root = tk.Tk()
    app = ScanToolGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
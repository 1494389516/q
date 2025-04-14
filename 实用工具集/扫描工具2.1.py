import nmap
import requests
from bs4 import BeautifulSoup
import argparse
import whois
import shodan
import sublist3r
import dns.resolver
import zapv2
import subprocess
import ssl
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import datetime
import firecrawl
import time
from typing import Optional
import urllib.parse 

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
    print(f"\n正在对 {url} 执行OWASP ZAP Web安全测试...")
    try:
        zap = zapv2.ZAPv2()
        print("正在启动ZAP扫描...")
        scan_id = zap.spider.scan(url)
        while int(zap.spider.status(scan_id)) < 100:
            print(f"爬虫进度: {zap.spider.status(scan_id)}%")
            time.sleep(5)
        print("爬虫完成，开始主动扫描...")
        scan_id = zap.ascan.scan(url)       
        print("\n[ZAP扫描结果]")
        alerts = zap.core.alerts()
        for alert in alerts:
            print(f"[+] {alert['alert']} (风险等级: {alert['risk']})")
            print(f"    URL: {alert['url']}")
            print(f"    描述: {alert['description']}")
            print("-" * 50)
        return alerts
    except Exception as e:
        print(f"ZAP扫描错误: {str(e)}")
        return None

def crawl_website(start_url: str, api_key: str, max_pages: Optional[int] = 10, allowed_domains: Optional[list] = None, scrape_formats: Optional[list] = None, poll_interval: int = 30):
    try:
        app = firecrawl.FirecrawlApp(api_key=api_key)
        
        params = {
            'limit': max_pages,
            'scrapeOptions': {
                'formats': scrape_formats if scrape_formats else ['markdown', 'html']
            }
        }
        if allowed_domains:
            params['allowedDomains'] = allowed_domains
        
        crawl_status = app.crawl_url(
            start_url,
            params=params,
            poll_interval=poll_interval
        )
        return crawl_status
    
    except firecrawl.exceptions.AuthenticationError:
        print("错误: API密钥无效")
        return None
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

def main():
    
    while True:
        print("\n扫描工具2.1")
        print("0. 退出程序")
        print("1. 端口扫描")
        print("2. WHOIS 查询")
        print("3. Shodan 查询")
        print("4. 全面扫描")
        print("5. 获取网页内容")
        print("6. 子域名枚举")
        print("7. DNS记录查询（含历史）")
        print("8. Web安全测试 (ZAP/SQLMap)")
        print("9. TheHarvester信息收集")
        print("10. SSL证书信息查询")
        print("11. Firecrawl网站爬取")
        print("12. 手工SQL注入和XSS测试")
        
        choice = input("请输入功能编号 (0-12): ")
        
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
            else:
                results = query_shodan(target, shodan_key)
                if results:
                    print("\n[Shodan查询结果]")
                    for item in results:
                        print(f"[+] {item}")
        elif choice == "4":
            target = input("请输入目标IP或域名: ")
            print(f"\n正在对 {target} 执行全面扫描...")
            # 端口扫描
            scan_type = input("请输入扫描类型 (syn 或 connect, 默认 syn): ") or "syn"
            results = scan_ports(target, scan_type)
            if results:
                print("\n[端口扫描结果]")
                for item in results:
                    print(f"[+] {item}")
            # WHOIS 查询
            results = query_whois(target)
            if results:
                print("\n[WHOIS查询结果]")
                for key, value in results.items():
                    print(f"{key}: {value}")
            # Shodan 查询
            shodan_key = input("请输入Shodan API密钥 (可选): ")
            if shodan_key:
                results = query_shodan(target, shodan_key)
                if results:
                    print("\n[Shodan查询结果]")
                    for item in results:
                        print(f"[+] {item}")
            else:
                print("\n[提示] 未提供Shodan API密钥，跳过Shodan查询")
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
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()





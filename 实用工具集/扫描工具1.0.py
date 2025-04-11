import nmap
import requests
from bs4 import BeautifulSoup
import argparse
import whois
import shodan
import sublist3r  # 确保 sublist3r.py 在 Python 的路径中
import dns.resolver
import zapv2
import subprocess
import ssl
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import datetime

def get_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
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
        
        # 提取SAN扩展
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
        
        # 历史查询（示例使用第三方API，需要替换为真实API）
        if history:
            print("正在查询历史解析记录...")
            # 这里需要实现具体的历史记录查询逻辑
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
            print(f"    描述: {alert['description']}")
            print("-" * 50)
        return alerts
    except Exception as e:
        print(f"ZAP扫描错误: {str(e)}")
        return None

def is_cert_valid(ssl_info):
    """检查SSL证书是否有效"""
    try:
        valid_from = datetime.datetime.strptime(ssl_info['有效期从'], '%Y-%m-%d')
        valid_to = datetime.datetime.strptime(ssl_info['有效期至'], '%Y-%m-%d')
        now = datetime.datetime.now()
        return valid_from <= now <= valid_to
    except Exception as e:
        print(f"证书有效性检查错误: {str(e)}")
        return False

def extract_domain(url):
    """从URL中提取域名"""
    return url.replace('https://', '').replace('http://', '').split('/')[0]

def main():
    
    while True:
        print("扫描工具1.0")
        print("0. 退出程序")
        print("1. 端口扫描")
        print("2. WHOIS 查询")
        print("3. Shodan 查询")
        print("4. 全面扫描")
        print("5. 获取网页内容")
        print("6. 子域名枚举")
        print("7. DNS记录查询（含历史）")
        print("8. OWASP ZAP Web安全测试")
        print("9. TheHarvester信息收集")
        print("10. SSL证书信息查询")
        
        choice = input("请输入功能编号 (0-10): ")
        
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
        elif choice == "5":
            url = input("请输入目标URL: ")
            html_content = get_web_page(url)
            if html_content:
                links = extract_links(html_content)
                print("\n[提取的链接]")
                for link in links:
                    print(f"[+] {link}")
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
            zap_scan(url)
        elif choice == "9":
            domain = input("请输入目标域名: ")
            results = theharvester(domain)
            if results:
                print("\n[TheHarvester结果]")
                print(results)
        elif choice == "10":
            url = input("请输入目标URL: ")
            print("\n[SSL证书检查]")
            ssl_info = get_ssl_info(url)
            
            if ssl_info:
                print("\n[SSL证书信息]")
                for key, value in ssl_info.items():
                    print(f"{key}: {value}")
                
                if is_cert_valid(ssl_info):
                    print("\n[✓] SSL证书有效")
                    print("\n[+] 正在自动启动子域名收集...")
                    domain = extract_domain(url)
                    Sublist3r(domain)
                else:
                    print("\n[!] SSL证书已过期或尚未生效")
                    proceed = input("是否仍要进行子域名收集？(y/n): ")
                    if proceed.lower() == 'y':
                        domain = extract_domain(url)
                        Sublist3r(domain)
            else:
                print("\n[!] 无法获取SSL证书信息")
                proceed = input("是否仍要进行子域名收集？(y/n): ")
                if proceed.lower() == 'y':
                    domain = extract_domain(url)
                    Sublist3r(domain)
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()





"""
Android network tools — ping, DNS, IP, port scan, Wi-Fi info.
Uses Linux networking commands available in Termux/Android.
"""

from __future__ import annotations

import subprocess
from langchain_core.tools import tool


def _shell_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out"
    except Exception as e:
        return f"[ERROR] {e}"


def _root_cmd(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            f"su -c '{cmd}'", shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def ping_host(host: str, count: int = 4) -> str:
    """فحص الاتصال بمضيف عبر ping.
    
    Args:
        host: اسم المضيف أو عنوان IP
        count: عدد المحاولات (افتراضي 4)
    """
    return _shell_cmd(f"ping -c {count} -W 5 {host}")


@tool
def get_network_info() -> str:
    """عرض معلومات الشبكة (IP، واجهات، DNS)."""
    info = []
    
    # IP addresses
    ip_info = _shell_cmd("ip addr show 2>/dev/null | grep 'inet ' | grep -v '127.0.0.1'")
    info.append(f"🌐 عناوين IP:\n{ip_info}")
    
    # Wi-Fi info
    wifi = _root_cmd("dumpsys wifi | grep 'Wi-Fi is' 2>/dev/null")
    ssid = _root_cmd("dumpsys wifi | grep 'mWifiInfo' 2>/dev/null | head -1")
    if wifi:
        info.append(f"\n📶 Wi-Fi: {wifi}")
    if ssid:
        info.append(f"📡 SSID: {ssid}")
    
    # DNS
    dns = _shell_cmd("getprop net.dns1 2>/dev/null")
    dns2 = _shell_cmd("getprop net.dns2 2>/dev/null")
    info.append(f"\n🔍 DNS: {dns}, {dns2}")
    
    # Default gateway
    gw = _shell_cmd("ip route | grep default")
    info.append(f"🚪 Gateway: {gw}")
    
    return "\n".join(info)


@tool
def get_public_ip() -> str:
    """عرض عنوان IP العام للجهاز."""
    result = _shell_cmd("curl -s --max-time 10 https://api.ipify.org 2>/dev/null")
    if result and "ERROR" not in result:
        return f"Public IP: {result}"
    result = _shell_cmd("curl -s --max-time 10 https://ifconfig.me 2>/dev/null")
    return f"Public IP: {result}" if result else "[ERROR] Could not determine public IP"


@tool
def dns_lookup(hostname: str) -> str:
    """البحث عن سجلات DNS لاسم مضيف.
    
    Args:
        hostname: اسم المضيف (مثل google.com)
    """
    # Try nslookup first, then dig, then host
    result = _shell_cmd(f"nslookup {hostname} 2>/dev/null")
    if not result or "ERROR" in result:
        result = _shell_cmd(f"dig +short {hostname} 2>/dev/null")
    if not result or "ERROR" in result:
        result = _shell_cmd(f"host {hostname} 2>/dev/null")
    if not result or "ERROR" in result:
        result = _shell_cmd(f"getent hosts {hostname} 2>/dev/null")
    return result or f"[ERROR] DNS lookup failed for {hostname}"


@tool
def check_port(host: str, port: int, timeout: int = 5) -> str:
    """فحص ما إذا كان منفذ TCP مفتوحاً.
    
    Args:
        host: عنوان المضيف
        port: رقم المنفذ
        timeout: مهلة الاتصال بالثواني (افتراضي 5)
    """
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return f"Port {port} on {host}: OPEN ✓"
        return f"Port {port} on {host}: CLOSED ✗"
    except socket.gaierror:
        return f"[ERROR] Cannot resolve hostname: {host}"
    except Exception as e:
        return f"[ERROR] {e}"


@tool
def wifi_scan() -> str:
    """فحص شبكات Wi-Fi المتاحة (يحتاج Root)."""
    result = _root_cmd("dumpsys wifi | grep 'SSID' | head -20")
    if not result or "ERROR" in result:
        result = _root_cmd("cmd wifi list-scan-results 2>/dev/null | head -20")
    return result or "[ERROR] Cannot scan Wi-Fi networks"


@tool
def traceroute(host: str) -> str:
    """تتبع مسار الاتصال إلى مضيف.
    
    Args:
        host: اسم المضيف أو عنوان IP
    """
    result = _shell_cmd(f"traceroute -m 15 -w 3 {host} 2>/dev/null")
    if not result or "ERROR" in result:
        result = _shell_cmd(f"tracepath {host} 2>/dev/null")
    return result or f"[ERROR] Traceroute failed for {host}"

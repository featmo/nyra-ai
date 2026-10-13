import datetime
from ddgs import DDGS
import psutil
import os
import platform
import subprocess
import sys
from PIL import ImageGrab

def get_date_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def web_search(query: str) -> str:
    try:
        print(f"Searching the web for you")
        results = DDGS().text(query, safesearch="off", max_results=5)
        if not results:
            return "No results found."
        return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)
    except Exception as e:
        return f"Search failed: {e}"

def web_fetch(url: str) -> str:
    try:
        page = DDGS().extract(url, fmt="text_plain")
        return page["content"][:4000]
    except Exception as e:
        return f"Could not read page: {e}"

def save_to_file(filename: str, content: str) -> str:
    try:
        with open(os.path.join(os.getcwd(), filename), "w", encoding="utf-8") as f:
            f.write(content)
        return f"Content saved to {filename}"
    except Exception as e:
        return f"Could not save to file: {e}"

def list_directory(path: str = ".") -> str:
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"Could not list directory: {e}"

def get_current_directory() -> str:
    return os.getcwd()

def get_file_contents(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Could not read file: {e}"


def list_processes() -> str:
    """ps-style process list: PID USER CPU% MEM% COMMAND (cross-platform)."""
    try:
        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
            try:
                info = p.info
                # try to get username
                try:
                    username = p.username()
                except Exception:
                    username = '?'
                cpu = info.get('cpu_percent', 0) or 0
                mem = info.get('memory_percent', 0) or 0
                cmd = ' '.join(info.get('cmdline') or [info.get('name', '')])
                procs.append((info.get('pid'), username, cpu, mem, cmd))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # sort by CPU desc
        procs.sort(key=lambda x: x[2], reverse=True)
        lines = ["PID      USER     CPU%  MEM%  COMMAND"]
        for pid, user, cpu, mem, cmd in procs[:50]:  # limit to top 50
            lines.append(f"{pid:<8} {user:<8} {cpu:>4.1f} {mem:>4.1f}  {cmd[:80]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Process list error: {e}"

def get_system_specs() -> str:
    """Concise system specs: OS, CPU, RAM, GPU."""
    try:
        os_name = platform.system()
        hostname = platform.node()
        cpu_cores = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        # GPU quick check: try nvidia-smi (Linux/Windows with drivers)
        gpu_str = "not detected"
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                gpu_str = res.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return (f"OS: {os_name} | Host: {hostname}\n"
                f"CPU: {cpu_cores} cores | Max freq: {freq.max if freq else 'N/A'} Hz\n"
                f"RAM: {mem.total / (2**30):.2f} GB total | {mem.free / (2**30):.2f} GB free | {mem.percent}% used\n"
                f"GPU: {gpu_str}")
    except Exception as e:
        return f"System specs error: {e}"

def screen_shot():
    # Source - https://stackoverflow.com/a/7045677
    # Posted by Gerrat, modified by community. See post 'Timeline' for change history
    # Retri0-13, License - CC BY-SA 3.0eved 2026-1
    try:
        im = ImageGrab.grab()
        path = "img.png"
        im.save(path,'PNG')
    except:
        print("no image")
    return path
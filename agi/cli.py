# filename: agi/watcher.py
import os
import sys
import time
import signal
import subprocess
import pathlib
import tty
import termios
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- 配置 ---
IGNORED_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".idea", ".pytest_cache", "dist", "build"}
WATCH_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".env"}
RESTART_DEBOUNCE_SECONDS = 1.5

TARGET_MODULE = "agi.console"

def reset_terminal():
    try:
        # 1. 基础 ANSI 重置
        sys.stdout.write("\033c\033[?25h\n")
        sys.stdout.flush()
        
        # 2. 【核心修复】强制恢复 TTY 的 cooked 模式
        # 这等同于在 shell 执行 'stty sane'
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        # 恢复回车换行映射 (ICRNL | ONLCR)
        attrs[0] |= termios.ICRNL  # 输入: 将回车映射为换行
        attrs[1] |= termios.ONLCR  # 输出: 将换行映射为回车换行
        attrs[3] |= (termios.ECHO | termios.ICANON | termios.ISIG) # 恢复回显和缓冲
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        
    except Exception as e:
        # 如果不是在交互式终端运行，tcgetattr 会报错，直接跳过即可
        pass

class ReloadHandler(FileSystemEventHandler):
    def __init__(self, module_name):
        self.module_name = module_name
        self.process = None
        self.last_restart = 0
        self.active = True
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.dirname(current_file_dir) 
        
        self.start_agent()

    def start_agent(self):
        if not self.active:
            return
        
        now = time.time()
        if now - self.last_restart < RESTART_DEBOUNCE_SECONDS:
            return
        self.last_restart = now

        if self.process:
            self._kill_process()

        if not self.active:
            return

        print(f"\n🔄 [{time.strftime('%H:%M:%S')}] 重启服务: python -m {self.module_name} ...")

        try:
            cmd = [sys.executable, "-m", self.module_name]
            
            self.process = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                env=os.environ,
                start_new_session=True, 
            )
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            self.active = False

    def _kill_process(self):
        if not self.process or self.process.poll() is not None:
            # 如果进程已经退出，也要执行一次终端重置，以防它是异常退出的
            if self.process and self.process.poll() is not None:
                reset_terminal()
            return

        print("⏹️  正在停止旧进程...")
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("⚠️  强制杀死进程组...")
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"⚠️ 清理进程出错: {e}")
            try:
                self.process.kill()
                self.process.wait()
            except:
                pass
        finally:
            # 【关键修复】无论进程如何结束，都尝试重置终端
            reset_terminal()

    def on_modified(self, event):
        if event.is_directory or not self.active:
            return
        
        src_path = pathlib.Path(event.src_path)
        if any(ignore in src_path.parts for ignore in IGNORED_DIRS):
            return
        if src_path.name.endswith(".tmp") or src_path.name == ".cli_session.json":
            return
        if src_path.suffix.lower() in WATCH_EXTENSIONS:
            self.start_agent()

    def check_process_status(self):
        """检查子进程是否意外或正常退出"""
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            # 如果 code 是 0，通常代表用户在子进程里输入了 exit 正常退出
            if code == 0:
                print(f"\n👋 子进程已正常退出。")
            else:
                print(f"\nℹ️ 子进程意外退出 (代码: {code})")
            
            reset_terminal()
            self.active = False
            return True # 返回 True 表示已经结束
        return False

if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(current_file_dir)
    os.chdir(work_dir)

    print(f"👀 热重载监控已启动 (-m 模式)")
    print(f"   包根目录: {work_dir}")
    print(f"   目标模块: {TARGET_MODULE}")
    print("   按 Ctrl+C 停止所有服务\n")

    handler = ReloadHandler(TARGET_MODULE)
    observer = Observer()
    observer.schedule(handler, path=".", recursive=True)
    observer.start()

    def signal_handler(sig, frame):
        print("\n🛑 接收到停止信号，正在关闭...")
        handler.active = False
        handler._kill_process() # 这里会调用 reset_terminal
        observer.stop()
        observer.join()
        
        # 最后一次确保终端干净
        reset_terminal()
        print("✅ 监控器已退出，终端已重置。")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            # 检查状态，如果返回 True 且 handler.active 为 False，说明子进程退出了
            if handler.check_process_status() or not handler.active:
                break
            time.sleep(0.5) # 稍微缩短检查频率，提升响应感
    except KeyboardInterrupt:
        pass
    finally:
        # 确保清理
        handler.active = False
        handler._kill_process()
        observer.stop()
        observer.join()
        reset_terminal()
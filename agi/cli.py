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
    """
    强制重置终端状态，解决子进程崩溃后导致的 bash 换行/显示混乱问题。
    发送 ANSI 重置序列并刷新 stdout。
    """
    try:
        # 1. 发送 "Reset Device" (RIS) 序列
        sys.stdout.write("\033c")
        # 2. 确保光标可见 (DECTCEM)
        sys.stdout.write("\033[?25h")
        # 3. 禁用鼠标跟踪 (如果之前开启了)
        sys.stdout.write("\033[?1000l\033[?1002l\033[?1003l")
        # 4. 退出备用屏幕缓冲区 (如果使用了)
        sys.stdout.write("\033[?1049l")
        # 5. 换行并刷新，确保提示符在新的一行
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        # 6. (可选) 尝试调用 tput reset，这更彻底但稍慢
        # os.system("tput reset > /dev/tty 2>&1") 
    except Exception:
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
        """检查子进程是否意外退出"""
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            print(f"\nℹ️  子进程意外退出 (代码: {code})")
            
            # 【关键修复】如果子进程自己崩了（比如报错退出），它可能没来得及恢复终端
            # 父进程检测到后，必须立即接管并重置终端
            reset_terminal()
            
            self.active = False

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
        while handler.active:
            time.sleep(1)
            handler.check_process_status()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        handler._kill_process()
        reset_terminal()
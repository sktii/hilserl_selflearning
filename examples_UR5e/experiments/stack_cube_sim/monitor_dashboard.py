import tkinter as tk
from tkinter import ttk
import requests
import time
import threading
import argparse

# 解析命令列參數
parser = argparse.ArgumentParser(description="Robot Monitor Dashboard")
parser.add_argument("--port", type=int, default=5000, help="Port of the robot server (default: 5000)")
args = parser.parse_args()

# 預設嘗試的埠號列表。優先嘗試 5001 (Actor)，然後是 5000 (Learner)
# 如果命令行有指定特定的 port，也會被考慮進去
TARGET_PORTS = [args.port, 5001, 5000]
# 去除重複並保持順序
TARGET_PORTS = list(dict.fromkeys(TARGET_PORTS))

class RobotMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hil-Serl Robot Monitor")
        self.root.geometry("400x350")
        self.root.configure(bg="#f0f0f0")

        # --- 修正 1: 刪除了這裡原本會報錯的 self.status_label.pack(...) ---

        self.session = requests.Session()
        
        # 標題
        title_label = tk.Label(root, text="Robot State Monitor", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        # 建立數據顯示區
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(padx=20, pady=5)

        self.labels = {}
        # 定義我們要監控的變數
        self.vars_to_monitor = [
            ("X", "0.000"), ("Y", "0.000"), ("Z", "0.000"),
            ("RX (qx)", "0.000"), ("RY (qy)", "0.000"), ("RZ (qz)", "0.000"), ("RW (qw)", "0.000"),
            ("Gripper", "0.000")
        ]

        for i, (name, init_val) in enumerate(self.vars_to_monitor):
            row = tk.Frame(self.frame, bg="#f0f0f0")
            row.pack(fill="x", pady=2)

            lbl_name = tk.Label(row, text=f"{name}:", width=10, anchor="w", font=("Consolas", 12), bg="#f0f0f0")
            lbl_name.pack(side="left")

            lbl_val = tk.Label(row, text=init_val, width=15, anchor="e", font=("Consolas", 12, "bold"), bg="white", fg="blue")
            lbl_val.pack(side="right")

            self.labels[name] = lbl_val

        # 狀態燈 (現在才建立，這是正確的位置)
        self.status_label = tk.Label(root, text="Connecting...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=5)

        self.running = True
        self.current_port_index = 0
        self.connected_port = None
        self.update_data()

    def update_data(self):
        if not self.running:
            return

        # 決定要連哪一個 Port
        if self.connected_port:
            target_port = self.connected_port
        else:
            target_port = TARGET_PORTS[self.current_port_index]

        # 建立當前的 URL
        server_url = f"http://127.0.0.1:{target_port}"

        try:
            # --- 修正 2: 這裡必須使用小寫的 server_url，否則永遠只會連到 5000 ---
            response = self.session.post(f"{server_url}/getstate", json={}, timeout=0.2)
            
            if response.status_code == 200:
                # 連接成功
                self.connected_port = target_port

                data = response.json()
                pose = data.get("pose", [0]*7) # [x, y, z, qx, qy, qz, qw]
                gripper = data.get("gripper_pos", 0)

                # 更新介面 (Pose)
                self.labels["X"].config(text=f"{pose[0]:.4f}")
                self.labels["Y"].config(text=f"{pose[1]:.4f}")
                self.labels["Z"].config(text=f"{pose[2]:.4f}")

                self.labels["RX (qx)"].config(text=f"{pose[3]:.4f}")
                self.labels["RY (qy)"].config(text=f"{pose[4]:.4f}")
                self.labels["RZ (qz)"].config(text=f"{pose[5]:.4f}")
                self.labels["RW (qw)"].config(text=f"{pose[6]:.4f}")

                # 更新介面 (Gripper)
                g_text = f"{gripper:.3f}"
                if gripper > 0.8: g_text += " (OPEN)"
                elif gripper < 0.1: g_text += " (CLOSED)"

                self.labels["Gripper"].config(text=g_text)

                self.status_label.config(text=f"● Connected to Port {target_port}", fg="green")
            else:
                self.status_label.config(text=f"Server Error: {response.status_code}", fg="red")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # 如果連線失敗且尚未鎖定 Port，則嘗試下一個 Port
            if not self.connected_port:
                self.status_label.config(text=f"Trying Port {target_port}...", fg="orange")
                self.current_port_index = (self.current_port_index + 1) % len(TARGET_PORTS)
            else:
                # 已經連上但突然斷線
                self.status_label.config(text="● Disconnected (Retrying...)", fg="red")
        
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            print(f"Dashboard Error: {str(e)}")

        # 每 50ms 更新一次 (20Hz)
        self.root.after(50, self.update_data)

    def on_closing(self):
        self.running = False
        self.session.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
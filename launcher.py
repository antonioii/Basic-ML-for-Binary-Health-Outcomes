###############################################################################
# Launcher to run backend and frontend servers automatically
###############################################################################

# --- Libs ---
import os
import subprocess
import threading
import webbrowser
import time
import signal
import sys
import tkinter as tk
from tkinter import ttk, messagebox

###############################################################################
# GENERAL CONFIGS
###############################################################################

# --- Folder (root) where this launcher is ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# --- Setting paths ---
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQ_FILE = os.path.join(PROJECT_ROOT, "backend", "requirements.txt")
ENV_LOCAL = os.path.join(PROJECT_ROOT, ".env.local")

# --- Backend ---
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = "8000"
BACKEND_ENTRY = "backend.main:app"

# --- Frontend ---
FRONTEND_PORT = "3000"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# --- API URL to frontend ---
VITE_API_URL_VALUE = f"http://localhost:{BACKEND_PORT}/api"

# --- Global process ---
backend_proc = None
frontend_proc = None


###############################################################################
# Help functions
###############################################################################

def create_or_update_env_file(api_key: str):
    """
    Create/update .env.local with Google API key (optional).
    """
    lines = [
        f"VITE_API_URL={VITE_API_URL_VALUE}",
        f"GEMINI_API_KEY={api_key.strip()}",
        "",
    ]
    with open(ENV_LOCAL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_cmd(cmd_list, cwd=None, use_shell=False):
    """
    Execute syncronus command and return (rc, stdout, stderr).
    use_shell=True to commands as 'npm ...' in Windows.
    """
    proc = subprocess.Popen(
        cmd_list if not use_shell else " ".join(cmd_list),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
        shell=use_shell,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def ensure_venv():
    """
    .venv with Python 3.11 (more stable).
    """
    if os.path.isdir(VENV_DIR):
        return True, "VENV já existe."
    rc, out, err = run_cmd(
        ["py", "-3.11", "-m", "venv", ".venv"],
        cwd=PROJECT_ROOT,
        use_shell=False
    )
    if rc != 0:
        return False, f"Falha ao criar venv:\n{err or out}"
    return True, "VENV criada com sucesso."


def pip_install_requirements(progress_callback=None):
    """
    Inside venv:
    1) python -m pip install --upgrade pip setuptools wheel
    2) pip install -r backend/requirements.txt
    """
    python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
    pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")

    steps = [
        [python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [pip_exe, "install", "-r", REQ_FILE],
    ]

    for step in steps:
        rc, out, err = run_cmd(step, cwd=PROJECT_ROOT)
        if progress_callback:
            if out:
                progress_callback(out)
            if err:
                progress_callback(err)
        if rc != 0:
            return False, f"Erro instalando dependências:\n{err or out}"

    return True, "Dependências backend instaladas."


def npm_install_if_needed(progress_callback=None):
    """
    Execute 'npm install' if node_modules/ doesn't exist.
    shell=True to found npm.cmd in Windows.
    """
    node_modules_dir = os.path.join(PROJECT_ROOT, "node_modules")
    if os.path.isdir(node_modules_dir):
        if progress_callback:
            progress_callback("[frontend] node_modules já existe, pulando npm install.")
        return True, "npm install pulado."

    rc, out, err = run_cmd(["npm", "install"], cwd=PROJECT_ROOT, use_shell=True)
    if progress_callback:
        if out:
            progress_callback(out)
        if err:
            progress_callback(err)
    if rc != 0:
        return False, f"Erro no npm install:\n{err or out}"

    return True, "Dependências frontend instaladas."


def start_backend():
    """
    Start backend (uvicorn) with venv.
    IMPORTANT: Do NOT capture stdout/stderr in PIPE in order to not buffer / block execution.
    """
    global backend_proc
    python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")

    cmd = [
        python_exe,
        "-m", "uvicorn",
        BACKEND_ENTRY,
        "--host", BACKEND_HOST,
        "--port", BACKEND_PORT,
        "--reload",
    ]

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    backend_proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,  # Do NOT block in PIPE
        stderr=subprocess.DEVNULL,  # Same
        text=True,
        creationflags=creationflags
    )

    # Short rest to aware crashes
    time.sleep(1.5)
    if backend_proc.poll() is not None:
        raise RuntimeError("Backend não conseguiu iniciar (processo terminou cedo).")

    return backend_proc


def start_frontend():
    """
    Roda `npm run dev` em processo separado.
    Também sem PIPE pra não travar.
    """
    global frontend_proc

    cmd = ["npm", "run", "dev"]
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    frontend_proc = subprocess.Popen(
        " ".join(cmd),
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        shell=True,
        creationflags=creationflags
    )

    # Short rest to aware crashes
    time.sleep(1.5)
    if frontend_proc.poll() is not None:
        raise RuntimeError("Frontend não conseguiu iniciar (processo terminou cedo).")

    return frontend_proc


def terminate_process(proc):
    """
    End open process.
    """
    if proc and proc.poll() is None:
        try:
            if sys.platform.startswith("win"):
                # CTRL_BREAK_EVENT only works if proc was created in "new process group"
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(0.5)
                except Exception:
                    pass
            proc.terminate()
            time.sleep(0.5)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


###############################################################################
# TKINTER GUI
###############################################################################

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Basic ML Launcher")
        self.geometry("480x360")
        self.resizable(False, False)

        # "start" -> ask API Key and Start
        # "running" -> End Application btn
        self.current_state = "start"

        self.frame_main = ttk.Frame(self, padding=20)
        self.frame_main.pack(fill="both", expand=True)

        self.log_text = tk.Text(self, height=7, state="disabled", wrap="word")
        self.log_text.pack(fill="both", padx=20, pady=(0, 10), expand=False)

        self.loading_label = ttk.Label(self, text="")
        self.loading_label.pack(pady=(0, 10))

        self.loading_running = False

        self.build_start_screen()

        self.protocol("WM_DELETE_WINDOW", self.on_close_request)

    def clear_frame(self):
        for child in self.frame_main.winfo_children():
            child.destroy()

    def append_log(self, msg: str):
        if not msg:
            return
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def set_loading(self, active: bool):
        self.loading_running = active
        if active:
            self.loading_label.config(text="⏳ Inicializando...")
            self.animate_loading()
        else:
            self.loading_label.config(text="")

    def animate_loading(self):
        if not self.loading_running:
            return
        current = self.loading_label.cget("text")
        frames = [
            "⏳ Inicializando...",
            "⏳ Inicializando.. ",
            "⏳ Inicializando.  ",
            "⏳ Inicializando   ",
        ]
        try:
            idx = frames.index(current)
            nxt = frames[(idx + 1) % len(frames)]
        except ValueError:
            nxt = frames[0]
        self.loading_label.config(text=nxt)
        self.after(400, self.animate_loading)

    def build_start_screen(self):
        self.current_state = "start"
        self.clear_frame()

        ttk.Label(
            self.frame_main,
            text="Basic ML Launcher",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(0, 10))

        ttk.Label(
            self.frame_main,
            text="Insira sua GEMINI_API_KEY (opcional) e clique em Start.\n"
                 "O app abrirá no navegador.",
            justify="center"
        ).pack(pady=(0, 20))

        self.api_var = tk.StringVar()
        row_frame = ttk.Frame(self.frame_main)
        row_frame.pack(pady=(0, 10), fill="x")

        ttk.Label(row_frame, text="GEMINI_API_KEY:").pack(anchor="w")
        self.api_entry = ttk.Entry(row_frame, textvariable=self.api_var, show="*")
        self.api_entry.pack(fill="x")

        self.start_btn = ttk.Button(
            self.frame_main,
            text="Start Execution",
            command=self.on_start_clicked
        )
        self.start_btn.pack(pady=(10, 5))

        ttk.Label(
            self.frame_main,
            text="Fluxo:\n1. Configura ambiente\n2. Sobe backend+frontend\n3. Abre navegador",
            justify="center",
            font=("Segoe UI", 9)
        ).pack(pady=(10, 0))

    def build_running_screen(self):
        self.current_state = "running"
        self.clear_frame()

        ttk.Label(
            self.frame_main,
            text="Aplicação em execução",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(0, 10))

        ttk.Label(
            self.frame_main,
            text=f"Use a aplicação em {FRONTEND_URL}.\n"
                 "Quando terminar clique em End Application.",
            justify="center"
        ).pack(pady=(0, 20))

        self.stop_btn = ttk.Button(
            self.frame_main,
            text="End Application",
            command=self.on_stop_clicked
        )
        self.stop_btn.pack(pady=(10, 5))

    def disable_start_ui(self):
        try:
            self.start_btn.config(state="disabled")
        except Exception:
            pass
        try:
            self.api_entry.config(state="disabled")
        except Exception:
            pass

    def enable_start_ui(self):
        try:
            self.start_btn.config(state="normal")
        except Exception:
            pass
        try:
            self.api_entry.config(state="normal")
        except Exception:
            pass

    def on_start_clicked(self):
        self.disable_start_ui()
        self.set_loading(True)
        self.append_log("Iniciando preparação...")
        th = threading.Thread(target=self.bootstrap_and_run, daemon=True)
        th.start()

    def bootstrap_and_run(self):
        # 1. .env.local
        api_value = ""
        try:
            api_value = self.api_var.get().strip()
        except Exception:
            pass
        create_or_update_env_file(api_value)
        self.append_log(".env.local atualizado.")

        # 2. venv
        ok, msg = ensure_venv()
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro criando venv", msg)
            return

        # 3. pip install
        ok, msg = pip_install_requirements(progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro instalando backend", msg)
            return

        # 4. npm install
        ok, msg = npm_install_if_needed(progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro instalando frontend", msg)
            return

        # 5. backend
        try:
            start_backend()
            self.append_log("Backend iniciado em http://localhost:8000")
        except Exception as e:
            self.safe_fail("Erro ao iniciar backend", str(e))
            return

        # 6. frontend
        try:
            start_frontend()
            self.append_log(f"Frontend iniciado em {FRONTEND_URL}")
        except Exception as e:
            # se frontend falhar, derruba backend também
            self.stop_processes()
            self.safe_fail("Erro ao iniciar frontend", str(e))
            return

        # 7. browser
        try:
            self.append_log("Aguarde 30s, servidores carregando...")
            time.sleep(30)
            webbrowser.open(FRONTEND_URL)
            self.append_log("Navegador aberto.")
        except Exception as e:
            self.append_log(f"Falha ao abrir navegador automaticamente: {e}")

        # 8. Ok
        self.after(0, self.to_running_state)

    def to_running_state(self):
        self.set_loading(False)
        self.build_running_screen()

    def safe_fail(self, title, msg):
        def _do():
            self.set_loading(False)
            self.enable_start_ui()
            messagebox.showerror(title, msg)
        self.after(0, _do)

    def on_stop_clicked(self):
        self.append_log("Encerrando aplicação...")
        self.stop_processes()
        self.append_log("Finalizado. Fechando janela.")
        self.after(300, self.destroy)

    def stop_processes(self):
        global backend_proc, frontend_proc
        try:
            terminate_process(frontend_proc)
        except Exception:
            pass
        try:
            terminate_process(backend_proc)
        except Exception:
            pass

    def on_close_request(self):
        if self.current_state == "running":
            if messagebox.askyesno(
                "Encerrar?",
                "A aplicação ainda está rodando.\n"
                "Deseja encerrar os servidores e fechar?"
            ):
                self.stop_processes()
                self.destroy()
        else:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()


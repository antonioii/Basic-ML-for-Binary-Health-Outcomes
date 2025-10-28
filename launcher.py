import os
import subprocess
import threading
import webbrowser
import time
import signal
import tkinter as tk
from tkinter import ttk, messagebox

###############################################################################
# CONFIGURAÇÕES GERAIS
###############################################################################

PROJECT_ROOT = os.path.abspath(os.getcwd())  # raiz do projeto
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQ_FILE = os.path.join(PROJECT_ROOT, "backend", "requirements.txt")
ENV_LOCAL = os.path.join(PROJECT_ROOT, ".env.local")

# Backend
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = "8000"
BACKEND_ENTRY = "backend.main:app"

# Frontend
FRONTEND_PORT = "3000"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# API URL que o frontend espera
VITE_API_URL_VALUE = f"http://localhost:{BACKEND_PORT}/api"

###############################################################################
# ESTADO GLOBAL DOS PROCESSOS
###############################################################################

backend_proc = None
frontend_proc = None


###############################################################################
# FUNÇÕES AUXILIARES
###############################################################################

def create_or_update_env_file(api_key: str):
    """
    Cria/atualiza .env.local com a API key.
    """
    lines = [
        f"VITE_API_URL={VITE_API_URL_VALUE}",
        f"GEMINI_API_KEY={api_key.strip()}",
        "",
    ]
    with open(ENV_LOCAL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_cmd(cmd_list, env=None, cwd=None, use_shell=False):
    """
    Executa comando síncrono e retorna (rc, stdout, stderr).
    use_shell=True para comandos tipo "npm ..." no Windows.
    """
    proc = subprocess.Popen(
        cmd_list if not use_shell else " ".join(cmd_list),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        text=True,
        shell=use_shell,
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def ensure_venv():
    """
    Garante .venv com Python 3.11.
    """
    if os.path.isdir(VENV_DIR):
        return True, "VENV já existe."
    rc, out, err = run_cmd(
        ["py", "-3.11", "-m", "venv", ".venv"],
        cwd=PROJECT_ROOT,
        use_shell=False
    )
    if rc != 0:
        return False, f"Falha ao criar venv:\n{err}"
    return True, "VENV criada com sucesso."


def pip_install_requirements(progress_callback=None):
    """
    Dentro da venv:
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
            progress_callback(out)
            progress_callback(err)
        if rc != 0:
            return False, f"Erro instalando dependências:\n{err}"
    return True, "Dependências instaladas."


def npm_install_if_needed(progress_callback=None):
    """
    Executa 'npm install' se node_modules não existir.
    Aqui usamos shell=True pra deixar o Windows resolver npm.cmd.
    """
    node_modules_dir = os.path.join(PROJECT_ROOT, "node_modules")
    if os.path.isdir(node_modules_dir):
        if progress_callback:
            progress_callback("node_modules já existe, pulando npm install.\n")
        return True, "npm install pulado."

    rc, out, err = run_cmd(["npm", "install"], cwd=PROJECT_ROOT, use_shell=True)
    if progress_callback:
        progress_callback(out)
        progress_callback(err)
    if rc != 0:
        return False, f"Erro no npm install:\n{err}"
    return True, "npm install concluído."


def start_backend():
    """
    Sobe backend (uvicorn) usando a venv.
    """
    global backend_proc
    python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")

    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        BACKEND_ENTRY,
        "--host",
        BACKEND_HOST,
        "--port",
        BACKEND_PORT,
        "--reload",
    ]

    backend_proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # ajuda matar depois
    )
    return backend_proc


def start_frontend():
    """
    Roda `npm run dev` em processo separado.
    shell=True no Windows pra localizar npm.cmd.
    """
    global frontend_proc

    cmd = ["npm", "run", "dev"]

    frontend_proc = subprocess.Popen(
        " ".join(cmd),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return frontend_proc


def terminate_process(proc):
    """
    Tenta encerrar um processo aberto via Popen no Windows.
    """
    if proc and proc.poll() is None:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            time.sleep(0.5)
        except Exception:
            pass
        try:
            proc.terminate()
            time.sleep(0.5)
        except Exception:
            pass
        try:
            if proc.poll() is None:
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

        # estados:
        # - "start": pedir API key e botão Start
        # - "running": botão End Application
        self.current_state = "start"

        # frame principal
        self.frame_main = ttk.Frame(self, padding=20)
        self.frame_main.pack(fill="both", expand=True)

        # campo de log
        self.log_text = tk.Text(self, height=7, state="disabled", wrap="word")
        self.log_text.pack(fill="both", padx=20, pady=(0, 10), expand=False)

        # label de loading animado
        self.loading_label = ttk.Label(self, text="")
        self.loading_label.pack(pady=(0, 10))

        self.loading_running = False  # controla animação "carregando..."

        self.build_start_screen()

        # fechar no X
        self.protocol("WM_DELETE_WINDOW", self.on_close_request)

    def clear_frame(self):
        for child in self.frame_main.winfo_children():
            child.destroy()

    def append_log(self, msg: str):
        if not msg:
            return
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def set_loading(self, active: bool):
        """
        Liga/desliga a animação "carregando..."
        """
        self.loading_running = active
        if active:
            self.loading_label.config(text="⏳ Inicializando...")
            self.animate_loading()
        else:
            self.loading_label.config(text="")

    def animate_loading(self):
        """
        Pisca/rotaciona o texto enquanto loading_running=True.
        """
        if not self.loading_running:
            return
        current = self.loading_label.cget("text")
        # Cria um efeito bobo girando pontinhos
        variants = [
            "⏳ Inicializando...",
            "⏳ Inicializando.. ",
            "⏳ Inicializando.  ",
            "⏳ Inicializando   ",
        ]
        try:
            idx = variants.index(current)
            nxt = variants[(idx + 1) % len(variants)]
        except ValueError:
            nxt = variants[0]
        self.loading_label.config(text=nxt)
        # agenda próxima troca
        self.after(400, self.animate_loading)

    def build_start_screen(self):
        self.current_state = "start"
        self.clear_frame()

        lbl_title = ttk.Label(
            self.frame_main,
            text="Basic ML Launcher",
            font=("Segoe UI", 14, "bold")
        )
        lbl_title.pack(pady=(0, 10))

        lbl_desc = ttk.Label(
            self.frame_main,
            text="Insira sua GEMINI_API_KEY (opcional) e clique em Start Execution.\n"
                 "A aplicação será iniciada no navegador.",
            justify="center"
        )
        lbl_desc.pack(pady=(0, 20))

        # Campo API Key
        self.api_var = tk.StringVar()
        row_frame = ttk.Frame(self.frame_main)
        row_frame.pack(pady=(0, 10), fill="x")
        ttk.Label(row_frame, text="GEMINI_API_KEY:").pack(anchor="w")
        self.api_entry = ttk.Entry(row_frame, textvariable=self.api_var, show="*")
        self.api_entry.pack(fill="x")

        # Botão Start
        self.start_btn = ttk.Button(
            self.frame_main,
            text="Start Execution",
            command=self.on_start_clicked
        )
        self.start_btn.pack(pady=(10, 5))

        # Info
        small_info = ttk.Label(
            self.frame_main,
            text="Ao iniciar:\n"
                 "1. Ambiente Python será configurado\n"
                 "2. Backend e Frontend serão iniciados\n"
                 "3. Seu navegador abrirá automaticamente",
            justify="center",
            font=("Segoe UI", 9)
        )
        small_info.pack(pady=(10, 0))

    def build_running_screen(self):
        self.current_state = "running"
        self.clear_frame()

        lbl_running = ttk.Label(
            self.frame_main,
            text="Aplicação em execução",
            font=("Segoe UI", 14, "bold")
        )
        lbl_running.pack(pady=(0, 10))

        lbl_hint = ttk.Label(
            self.frame_main,
            text="Use a aplicação no navegador.\n"
                 "Quando terminar, clique em End Application.",
            justify="center"
        )
        lbl_hint.pack(pady=(0, 20))

        self.stop_btn = ttk.Button(
            self.frame_main,
            text="End Application",
            command=self.on_stop_clicked
        )
        self.stop_btn.pack(pady=(10, 5))

    def disable_start_ui(self):
        """
        Enquanto carrega: bloqueia os inputs.
        """
        try:
            self.start_btn.config(state="disabled")
        except Exception:
            pass
        try:
            self.api_entry.config(state="disabled")
        except Exception:
            pass

    def enable_start_ui(self):
        """
        Se algo falhar e a gente quiser deixar tentar de novo.
        """
        try:
            self.start_btn.config(state="normal")
        except Exception:
            pass
        try:
            self.api_entry.config(state="normal")
        except Exception:
            pass

    def on_start_clicked(self):
        """
        Inicia a thread que faz todo o bootstrap.
        """
        self.disable_start_ui()
        self.set_loading(True)
        self.append_log("Iniciando preparação...")
        th = threading.Thread(target=self.bootstrap_and_run, daemon=True)
        th.start()

    def bootstrap_and_run(self):
        """
        Passo a passo:
        - cria/atualiza .env.local
        - garante venv
        - pip install
        - npm install se necessário
        - sobe backend e frontend
        - abre navegador
        - troca tela pra 'running'
        """
        # 1. .env.local
        try:
            api_value = self.api_var.get().strip()
        except Exception:
            api_value = ""
        create_or_update_env_file(api_value)
        self.append_log(".env.local atualizado (API Key opcional aplicada).")

        # 2. venv
        ok, msg = ensure_venv()
        self.append_log(msg)
        if not ok:
            self.append_log("Falha crítica ao criar venv. Abortando.")
            self.safe_fail("Erro", msg)
            return

        # 3. pip install
        ok, msg = pip_install_requirements(progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.append_log("Falha crítica ao instalar dependências. Abortando.")
            self.safe_fail("Erro em pip install", msg)
            return

        # 4. npm install (se precisar)
        ok, msg = npm_install_if_needed(progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.append_log("Falha crítica no npm install. Abortando.")
            self.safe_fail("Erro em npm install", msg)
            return

        # 5. backend
        try:
            start_backend()
            self.append_log("Backend iniciado em http://localhost:8000")
        except Exception as e:
            err_msg = f"Falha ao iniciar backend: {e}"
            self.append_log(err_msg)
            self.safe_fail("Erro ao iniciar backend", err_msg)
            return

        # 6. frontend
        try:
            start_frontend()
            self.append_log(f"Frontend iniciado em {FRONTEND_URL}")
        except Exception as e:
            err_msg = f"Falha ao iniciar frontend: {e}"
            self.append_log(err_msg)
            # mata backend antes de sair
            self.stop_processes()
            self.safe_fail("Erro ao iniciar frontend", err_msg)
            return

        # 7. abrir navegador
        try:
            webbrowser.open(FRONTEND_URL)
            self.append_log("Navegador aberto.")
        except Exception as e:
            self.append_log(f"Não consegui abrir navegador automaticamente: {e}")

        # 8. tudo ok -> estado running
        self.after(0, self.to_running_state)

    def to_running_state(self):
        """
        Chamado só se tudo deu certo.
        """
        self.set_loading(False)
        self.build_running_screen()

    def safe_fail(self, title, msg):
        """
        Chamado quando algo deu errado no bootstrap.
        Volta pra tela inicial "tentar de novo".
        """
        def _do():
            self.set_loading(False)
            self.enable_start_ui()
            messagebox.showerror(title, msg)
            # não mata backend aqui porque se deu erro, backend provavelmente nem subiu
        self.after(0, _do)

    def on_stop_clicked(self):
        """
        Botão End Application
        """
        self.append_log("Encerrando aplicação...")
        self.stop_processes()
        self.append_log("Finalizado. Fechando janela.")
        # pequena pausa só pra mensagem aparecer
        self.after(300, self.destroy)

    def stop_processes(self):
        """
        Encerra backend e frontend.
        """
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
        """
        Clicar no X da janela.
        Se já está rodando, pergunta se quer encerrar tudo.
        """
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

    def safe_show_error(self, title, msg):
        """
        (mantido, mas agora usamos safe_fail)
        """
        def _show():
            messagebox.showerror(title, msg)
        self.after(0, _show)


if __name__ == "__main__":
    app = App()
    app.mainloop()

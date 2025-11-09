###############################################################################
# Launcher to run backend and frontend servers automatically
###############################################################################

# --- Libs ---
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

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
SUPPORTED_SYSTEMS = ("windows", "linux")
FRONTEND_INSTALL_MARKER = os.path.join(PROJECT_ROOT, ".frontend_deps_ready")
MIN_NPM_MAJOR = 8
PATCH_PACKAGE_ERROR_TOKENS = ("patch-package",)


def normalize_system_choice(choice: str) -> str:
    """
    Normalize UI input to a supported system string.
    """
    if not choice:
        return "windows"
    choice = choice.lower()
    return choice if choice in SUPPORTED_SYSTEMS else "windows"


def get_scripts_dir(system_choice: str) -> str:
    return "Scripts" if system_choice == "windows" else "bin"


def get_python_executable(system_choice: str) -> str:
    exe = "python.exe" if system_choice == "windows" else "python"
    return os.path.join(VENV_DIR, get_scripts_dir(system_choice), exe)


def get_pip_executable(system_choice: str) -> str:
    exe = "pip.exe" if system_choice == "windows" else "pip"
    return os.path.join(VENV_DIR, get_scripts_dir(system_choice), exe)


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


def parse_semver_major(version: str):
    """
    Extract major component from semver strings such as '9.2.0' or 'v9.2.0'.
    """
    if not version:
        return None
    token = version.strip()
    if token.startswith(("v", "V")):
        token = token[1:]
    try:
        return int(token.split(".")[0])
    except (ValueError, IndexError):
        return None


def diagnose_node_environment(progress_callback=None):
    """
    Log Node/NPM versions and binary locations (mirrors a-fazer checklist).
    """
    locator = "where" if sys.platform.startswith("win") else "which"
    commands = [
        ("node -v", ["node", "-v"]),
        ("npm -v", ["npm", "-v"]),
        (f"{locator} node", [locator, "node"]),
        (f"{locator} npm", [locator, "npm"]),
    ]
    detected_npm = None
    for label, cmd in commands:
        rc, out, err = run_cmd(cmd, cwd=PROJECT_ROOT, use_shell=True)
        text = (out or err or f"(exit code {rc})").strip()
        if progress_callback:
            progress_callback(f"{label}: {text}")
        if label == "npm -v" and rc == 0 and text:
            detected_npm = text.splitlines()[-1]
    return detected_npm


def ensure_npm_version(progress_callback=None, detected_version=None):
    """
    Guarantee npm >= MIN_NPM_MAJOR; attempt auto-upgrade when outdated.
    """
    version = detected_version
    if not version:
        rc, out, err = run_cmd(["npm", "-v"], cwd=PROJECT_ROOT, use_shell=True)
        if rc != 0:
            return False, f"npm não encontrado ({err or out}). Instale Node.js antes de continuar."
        version = (out or err or "").strip().splitlines()[-1]

    major = parse_semver_major(version)
    if major is None:
        return False, f"Não foi possível interpretar a versão do npm: '{version}'. Atualize manualmente."

    if major >= MIN_NPM_MAJOR:
        if progress_callback:
            progress_callback(f"npm versão detectada: {version} (ok para o launcher).")
        return True, version

    if progress_callback:
        progress_callback(
            f"npm versão {version} detectada (< {MIN_NPM_MAJOR}). Tentando atualizar com 'npm install -g npm@latest'..."
        )
    rc, out, err = run_cmd(["npm", "install", "-g", "npm@latest"], cwd=PROJECT_ROOT, use_shell=True)
    if progress_callback:
        if out:
            progress_callback(out)
        if err:
            progress_callback(err)

    rc_ver, out_ver, err_ver = run_cmd(["npm", "-v"], cwd=PROJECT_ROOT, use_shell=True)
    if rc_ver != 0:
        return False, "Falha ao verificar npm após tentativa de atualização. Execute 'npm install -g npm@latest' manualmente."
    new_version = (out_ver or err_ver or "").strip().splitlines()[-1]
    new_major = parse_semver_major(new_version)
    if new_major is not None and new_major >= MIN_NPM_MAJOR:
        if progress_callback:
            progress_callback(f"npm atualizado para {new_version}.")
        return True, new_version

    return False, (
        f"npm continua na versão {new_version} (< {MIN_NPM_MAJOR}). "
        "Siga o plano do arquivo a-fazer.txt (instale Node LTS ou ajuste via nvm) antes de reiniciar o launcher."
    )


def clean_frontend_artifacts(progress_callback=None):
    """
    Remove node_modules, package-lock.json, and the install marker.
    """
    removed = []
    node_modules_dir = os.path.join(PROJECT_ROOT, "node_modules")
    if os.path.isdir(node_modules_dir):
        shutil.rmtree(node_modules_dir, ignore_errors=True)
        removed.append("node_modules")

    lock_file = os.path.join(PROJECT_ROOT, "package-lock.json")
    if os.path.isfile(lock_file):
        try:
            os.remove(lock_file)
            removed.append("package-lock.json")
        except OSError:
            pass

    if os.path.isfile(FRONTEND_INSTALL_MARKER):
        try:
            os.remove(FRONTEND_INSTALL_MARKER)
            removed.append(os.path.basename(FRONTEND_INSTALL_MARKER))
        except OSError:
            pass

    if progress_callback:
        if removed:
            progress_callback(f"[frontend] Limpeza executada ({', '.join(removed)}).")
        else:
            progress_callback("[frontend] Nada para limpar.")


def mark_frontend_install_success():
    """
    Drop a tiny marker file to avoid repeated npm install runs.
    """
    try:
        Path(FRONTEND_INSTALL_MARKER).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def should_skip_npm_install():
    node_modules_dir = os.path.join(PROJECT_ROOT, "node_modules")
    return os.path.isdir(node_modules_dir) and os.path.isfile(FRONTEND_INSTALL_MARKER)


def install_patch_package_temp(progress_callback=None):
    """
    Ensure patch-package CLI is available (addresses npm postinstall failures).
    """
    cmd = ["npm", "install", "patch-package", "--no-save"]
    rc, out, err = run_cmd(cmd, cwd=PROJECT_ROOT, use_shell=True)
    if progress_callback:
        if out:
            progress_callback(out)
        if err:
            progress_callback(err)
    if rc != 0:
        return False, "Falha ao instalar patch-package automaticamente. Instale manualmente com 'npm install -D patch-package'."
    return True, "patch-package disponível."


def ensure_venv(system_choice: str):
    """
    Create .venv using the interpreter that matches the selected OS.
    """
    if os.path.isdir(VENV_DIR):
        return True, "VENV já existe."
    if system_choice == "windows":
        cmd = ["py", "-3.11", "-m", "venv", ".venv"]
    else:
        cmd = ["python3", "-m", "venv", ".venv"]
    rc, out, err = run_cmd(
        cmd,
        cwd=PROJECT_ROOT,
        use_shell=False
    )
    if rc != 0:
        return False, f"Falha ao criar venv:\n{err or out}"
    return True, "VENV criada com sucesso."


def pip_install_requirements(system_choice: str, progress_callback=None):
    """
    Inside venv:
    1) python -m pip install --upgrade pip setuptools wheel
    2) pip install -r backend/requirements.txt
    """
    python_exe = get_python_executable(system_choice)
    pip_exe = get_pip_executable(system_choice)

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


def npm_install_if_needed(progress_callback=None, force=False):
    """
    Install frontend dependencies with resilience (clean + retry + patch-package support).
    """
    if not force and should_skip_npm_install():
        if progress_callback:
            progress_callback("[frontend] Dependências já instaladas (marcador encontrado).")
        return True, "npm install pulado."

    attempts = 0
    clean_attempted = False
    while attempts < 3:
        attempts += 1
        rc, out, err = run_cmd(["npm", "install"], cwd=PROJECT_ROOT, use_shell=True)
        if progress_callback:
            if out:
                progress_callback(out)
            if err:
                progress_callback(err)

        if rc == 0:
            mark_frontend_install_success()
            return True, "Dependências frontend instaladas."

        message = (err or out or "").strip()
        if progress_callback:
            progress_callback(f"[frontend] npm install falhou (tentativa {attempts}).")

        lower_message = message.lower()
        if any(token in lower_message for token in PATCH_PACKAGE_ERROR_TOKENS):
            if progress_callback:
                progress_callback("[frontend] patch-package não encontrado. Instalando antes de tentar novamente...")
            ok_patch, info_patch = install_patch_package_temp(progress_callback)
            if not ok_patch:
                return False, info_patch
            clean_frontend_artifacts(progress_callback)
            continue

        if not clean_attempted:
            clean_frontend_artifacts(progress_callback)
            clean_attempted = True
            continue

        return False, f"Erro no npm install:\n{message or 'Consulte os logs acima.'}"

    return False, "npm install falhou após múltiplas tentativas."


def prepare_frontend_environment(progress_callback=None):
    """
    Execute the checklist from a-fazer.txt prior to npm install.
    """
    if progress_callback:
        progress_callback("Verificando ambiente Node/NPM (passo 1 do a-fazer.txt)...")
    detected_npm = diagnose_node_environment(progress_callback)

    ok, msg = ensure_npm_version(progress_callback, detected_version=detected_npm)
    if not ok:
        return False, msg

    if progress_callback:
        progress_callback("Garantindo dependências do frontend (incluindo patch-package).")
    return npm_install_if_needed(progress_callback)


def start_backend(system_choice: str):
    """
    Start backend (uvicorn) with venv.
    IMPORTANT: Do NOT capture stdout/stderr in PIPE in order to not buffer / block execution.
    """
    global backend_proc
    python_exe = get_python_executable(system_choice)

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
    No Windows: cria um novo "process group" e usa shell=True.
    No Linux/WSL: inicia sem shell e cria um "process group" POSIX (setsid),
    permitindo matar todos os filhos (npm, node, vite).
    """
    global frontend_proc

    cmd = ["npm", "run", "dev"]

    if sys.platform.startswith("win"):
        # Windows
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
    else:
        # Linux/WSL
        frontend_proc = subprocess.Popen(
            cmd,                             # sem shell para não criar camada extra
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid             # novo grupo de processos (POSIX)
        )

    time.sleep(1.5)
    if frontend_proc.poll() is not None:
        raise RuntimeError("Frontend não conseguiu iniciar (processo terminou cedo).")

    return frontend_proc


def terminate_process(proc):
    """
    Encerra um processo e seu grupo (quando aplicável).
    - Windows: envia CTRL_BREAK_EVENT e depois terminate/kill.
    - Linux/WSL: envia SIGTERM ao grupo; se resistir, SIGKILL.
    """
    if not proc or proc.poll() is not None:
        return

    try:
        if sys.platform.startswith("win"):
            # precisa ter sido criado com CREATE_NEW_PROCESS_GROUP
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.6)
            except Exception:
                pass
            try:
                proc.terminate()
                time.sleep(0.6)
            except Exception:
                pass
        else:
            # matar TODO o grupo (npm + node + vite)
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(0.8)
            except Exception:
                pass

        # se ainda estiver vivo, força
        if proc.poll() is None:
            if sys.platform.startswith("win"):
                try:
                    proc.kill()
                except Exception:
                    pass
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass



###############################################################################
# TKINTER GUI
###############################################################################

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Basic ML Launcher")
        self.geometry("480x600")
        self.resizable(False, False)

        # "start" -> ask API Key and Start
        # "running" -> End Application btn
        self.current_state = "start"
        self.system_var = tk.StringVar(value="windows")

        self.frame_main = ttk.Frame(self, padding=20)
        self.frame_main.pack(fill="both", expand=True)

        self.log_text = tk.Text(self, height=7, state="disabled", wrap="word")
        self.log_text.pack(fill="both", padx=20, pady=(0, 10), expand=False)

        self.loading_label = ttk.Label(self, text="")
        self.loading_label.pack(pady=(0, 10))

        self.loading_running = False
        self.os_radios = []

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
                 "O app abrirá no navegador ao final.\n No primeiro uso pode demorar alguns minutos!",
            justify="center"
        ).pack(pady=(0, 20))

        self.api_var = tk.StringVar()
        row_frame = ttk.Frame(self.frame_main)
        row_frame.pack(pady=(0, 10), fill="x")

        ttk.Label(row_frame, text="GEMINI_API_KEY:").pack(anchor="w")
        self.api_entry = ttk.Entry(row_frame, textvariable=self.api_var, show="*")
        self.api_entry.pack(fill="x")

        system_frame = ttk.LabelFrame(
            self.frame_main,
            text="Sistema operacional:"
        )
        system_frame.pack(fill="x", pady=(10, 10))
        win_radio = ttk.Radiobutton(
            system_frame,
            text="Windows System",
            variable=self.system_var,
            value="windows"
        )
        win_radio.pack(anchor="w", pady=2)
        linux_radio = ttk.Radiobutton(
            system_frame,
            text="Linux System",
            variable=self.system_var,
            value="linux"
        )
        linux_radio.pack(anchor="w", pady=2)
        self.os_radios = [win_radio, linux_radio]

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
        for radio in getattr(self, "os_radios", []):
            try:
                radio.config(state="disabled")
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
        for radio in getattr(self, "os_radios", []):
            try:
                radio.config(state="normal")
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
        system_choice = "windows"
        try:
            system_choice = normalize_system_choice(self.system_var.get())
        except Exception:
            pass
        self.append_log(f"Sistema selecionado: {'Windows System' if system_choice == 'windows' else 'Linux System'}.")
        create_or_update_env_file(api_value)
        self.append_log(".env.local atualizado.")

        # 2. venv
        ok, msg = ensure_venv(system_choice)
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro criando venv", msg)
            return

        # 3. pip install
        ok, msg = pip_install_requirements(system_choice, progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro instalando backend", msg)
            return

        # 4. npm install (com diagnóstico do a-fazer)
        ok, msg = prepare_frontend_environment(progress_callback=self.append_log)
        self.append_log(msg)
        if not ok:
            self.safe_fail("Erro instalando frontend", msg)
            return

        # 5. backend
        try:
            start_backend(system_choice)
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

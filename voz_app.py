#!/usr/bin/env python3
"""
Voz -> Texto  (ditado por voz, offline)

- Captura o microfone no canal 1 da Scarlett 18i20 (extrai o c0 de 18 canais)
- Transcreve com whisper.cpp (modelo large-v3-turbo) na GPU via Vulkan
- O modelo fica carregado na VRAM (whisper-server), entao cada ditado e' rapido
- Interface: botao de microfone (clique p/ ouvir, clique p/ parar), area de texto,
  botoes Copiar / Limpar. Idioma travado em PT-BR.

Instancia unica: abrir de novo (Meta+H) apenas mostra/foca a janela existente.
Fechar a janela apenas esconde (o modelo segue quente). "Sair" encerra de vez.
"""
import os
import sys
import math
import time
import signal
import socket
import tempfile
import json
import threading
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF
from PySide6.QtGui import (QFont, QPainter, QColor, QPen, QBrush, QIcon,
                           QGuiApplication, QAction, QPixmap, QTextCursor)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QTextEdit, QLabel, QSystemTrayIcon,
                               QMenu)
from PySide6.QtNetwork import QLocalServer, QLocalSocket

import requests

# ----------------------------- configuracao --------------------------------
HOME = Path.home()
BASE = Path(__file__).resolve().parent      # raiz do projeto (onde esta o app)
WHISPER_DIR = BASE / "whisper.cpp"

APP_NAME = "NEUROVOX"
HOST = "127.0.0.1"
PORT = 8917
SOCKET_NAME = "voz-stt-singleton"
DIAG = BASE / "diag.log"
SERVER_LOG = BASE / "whisper-server.log"


def _load_config():
    """Le config.json (ao lado do app) por cima dos padroes. Tudo opcional."""
    cfg = {
        "language": "pt", "model": "large-v3-turbo", "engine": "cpu",
        "device": "default", "channels": 1, "channel": 0,
        "rate": 48000, "threads": 8,
    }
    try:
        cfg.update(json.loads((BASE / "config.json").read_text()))
    except Exception:
        pass
    return cfg


CFG = _load_config()
LANG = CFG["language"]
MODEL = WHISPER_DIR / "models" / f"ggml-{CFG['model']}.bin"
# engine: "gpu" usa o build Vulkan (build/); qualquer outro usa CPU (build-cpu/)
_ENGINE_DIR = "build" if CFG["engine"] == "gpu" else "build-cpu"
WHISPER_SERVER = WHISPER_DIR / _ENGINE_DIR / "bin" / "whisper-server"
DEVICE = CFG["device"]              # "default" = mic padrao do sistema
REC_CHANNELS = int(CFG["channels"])
MIC_CHANNEL = int(CFG["channel"])
REC_RATE = int(CFG["rate"])
THREADS = str(CFG["threads"])


def resolve_device():
    """(target, channels) para o pw-record. target=None => mic padrao do sistema."""
    if DEVICE and DEVICE != "default":
        return DEVICE, REC_CHANNELS
    return None, REC_CHANNELS


def channel_levels(raw_path, channels):
    """RMS (dB) de cada canal do PCM bruto. Retorna lista [(idx, db), ...]."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", raw_path,
             "-af", "astats=metadata=1:reset=0", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return []
    res, cur = {}, None
    for ln in p.stderr.splitlines():
        if "Channel:" in ln:
            try:
                cur = int(ln.split("Channel:")[1].strip()) - 1  # 1-based -> 0-based
            except ValueError:
                cur = None
        elif "RMS level dB" in ln and cur is not None:
            val = ln.split(":")[-1].strip()
            try:
                res[cur] = float(val)
            except ValueError:
                res[cur] = -120.0
            cur = None
    return [(i, res.get(i, -120.0)) for i in range(channels)]


def pick_mic_channel(raw_path, channels):
    """Escolhe o canal com mais sinal (o mic). Registra os niveis no diag.log."""
    levels = channel_levels(raw_path, channels)
    chan = MIC_CHANNEL
    if levels:
        idx, db = max(levels, key=lambda x: x[1])
        if db > -75:                      # bem acima do piso de ruido (~-95)
            chan = idx
        try:
            with open(DIAG, "a") as f:
                f.write(" ".join(f"c{i}={d:.0f}" for i, d in levels) +
                        f"  -> c{chan} ({db:.0f} dB)\n")
        except OSError:
            pass
    return chan


# ----------------------------- whisper-server ------------------------------
class WhisperServer:
    def __init__(self):
        self.proc = None

    def already_running(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            s.connect((HOST, PORT))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def start(self):
        if self.already_running():
            return
        # log do servidor em arquivo (antes ia p/ /dev/null e mascarava falhas)
        try:
            log = open(SERVER_LOG, "a", buffering=1)
        except OSError:
            log = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [str(WHISPER_SERVER), "-m", str(MODEL), "-l", LANG, "-nt",
             "-t", THREADS, "--host", HOST, "--port", str(PORT)],
            stdout=log, stderr=log,
        )

    def ensure_running(self):
        """Garante o motor de pe; ressuscita se tiver caido (suspend, OOM…)."""
        if self.already_running():
            return            # porta aberta = vivo
        if self.proc and self.proc.poll() is None:
            return            # processo vivo, ainda carregando o modelo
        self.proc = None
        self.start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ----------------------------- threads -------------------------------------
class ReadyThread(QThread):
    """Espera o servidor (modelo) ficar pronto."""
    ready = Signal()

    def run(self):
        while True:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            try:
                s.connect((HOST, PORT))
                s.close()
                self.ready.emit()
                return
            except OSError:
                s.close()
                self.msleep(300)


# serializa as requisicoes ao whisper-server: 1 inferencia por vez.
# previne o crash do servidor por requisicoes concorrentes (parcial + final).
_server_lock = threading.Lock()


def transcribe_buffer(raw_path, channels):
    """Extrai o canal do mic do PCM cru, normaliza e transcreve no whisper-server.
    Usa arquivo temp unico p/ poder rodar passadas simultaneas sem colidir."""
    fd, mono = tempfile.mkstemp(prefix="voz-m-", suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(REC_RATE), "-ac", str(channels),
             "-i", raw_path,
             "-af", f"pan=mono|c0=c{MIC_CHANNEL},highpass=f=70,dynaudnorm=g=7",
             "-ar", "16000", "-ac", "1", mono],
            check=True,
        )
        with _server_lock:  # 1 requisicao por vez ao motor
            with open(mono, "rb") as f:
                r = requests.post(
                    f"http://{HOST}:{PORT}/inference",
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"response_format": "text", "language": LANG,
                          "temperature": "0", "no_timestamps": "true"},
                    timeout=120,
                )
            r.raise_for_status()
            text = r.text.strip()
    finally:
        try:
            os.remove(mono)
        except OSError:
            pass
    for junk in ("[BLANK_AUDIO]", "[SILENCE]", "(silence)", "[ Silence ]"):
        text = text.replace(junk, "")
    return text.strip()


class Worker(QThread):
    """Transcreve o buffer atual. final=True => passada definitiva (no 'parar')."""
    done = Signal(str, bool)
    failed = Signal(str, bool)

    def __init__(self, raw_path, channels, final):
        super().__init__()
        self.raw_path = raw_path
        self.channels = channels
        self.final = final

    def run(self):
        try:
            text = transcribe_buffer(self.raw_path, self.channels)
        except subprocess.CalledProcessError as e:
            self.failed.emit(f"Falha ao converter audio: {e}", self.final)
            return
        except Exception as e:
            self.failed.emit(f"Falha na transcricao: {e}", self.final)
            return
        self.done.emit(text, self.final)


# ----------------------------- botao de microfone --------------------------
class MicButton(QPushButton):
    IDLE, RECORDING, BUSY, LOADING = range(4)

    def __init__(self):
        super().__init__()
        self.state = self.LOADING
        self.setFixedSize(150, 150)
        self.setCursor(Qt.PointingHandCursor)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_state(self, state):
        self.state = state
        if state == self.RECORDING:
            self._timer.start(40)
        else:
            self._timer.stop()
            self._phase = 0.0
        self.setEnabled(state in (self.IDLE, self.RECORDING))
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        cx, cy = rect.center().x(), rect.center().y()
        r = 58

        colors = {
            self.IDLE:      QColor("#7c5cff"),
            self.RECORDING: QColor("#ff4d5e"),
            self.BUSY:      QColor("#5a5a66"),
            self.LOADING:   QColor("#3a3a44"),
        }
        col = colors[self.state]

        # anel pulsante durante a gravacao
        if self.state == self.RECORDING:
            pulse = (math.sin(self._phase) + 1) / 2
            ring_r = r + 8 + pulse * 16
            ring = QColor(col)
            ring.setAlphaF(0.18 + 0.18 * (1 - pulse))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(ring))
            p.drawEllipse(QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2))

        # circulo principal
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(col))
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # icone
        p.setPen(QPen(QColor("white"), 5, Qt.SolidLine, Qt.RoundCap))
        if self.state == self.RECORDING:
            # quadrado de "stop"
            s = 26
            p.setBrush(QBrush(QColor("white")))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(cx - s / 2, cy - s / 2, s, s), 5, 5)
        else:
            # microfone
            p.setBrush(QBrush(QColor("white")))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(cx - 11, cy - 26, 22, 36), 11, 11)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("white"), 5, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(cx - 20, cy - 18, 40, 40), 200 * 16, 140 * 16)
            p.drawLine(int(cx), int(cy + 18), int(cx), int(cy + 28))
            p.drawLine(int(cx - 12), int(cy + 28), int(cx + 12), int(cy + 28))
        p.end()


# ----------------------------- janela --------------------------------------
class MainWindow(QWidget):
    def __init__(self, server: WhisperServer):
        super().__init__()
        self.server = server
        self.target, self.channels = resolve_device()
        self.rec_proc = None
        self.raw_path = None
        self._really_quit = False

        # transcricao ao vivo (streaming)
        self._base = ""
        self._stopping = False
        self._partial_busy = False
        self._retried = False
        self.partial_worker = None
        self.final_worker = None
        self.partial_timer = QTimer(self)
        self.partial_timer.setInterval(1700)
        self.partial_timer.timeout.connect(self.on_partial_tick)

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(560, 560)
        self.setWindowIcon(make_icon())

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel("DITADO NEURAL · PT-BR · OFFLINE")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        self.mic = MicButton()
        self.mic.clicked.connect(self.toggle_record)
        mic_row = QHBoxLayout()
        mic_row.addStretch()
        mic_row.addWidget(self.mic)
        mic_row.addStretch()
        root.addLayout(mic_row)

        self.status = QLabel("Carregando modelo na GPU…")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status)

        self.text = QTextEdit()
        self.text.setPlaceholderText("O texto ditado aparece aqui…")
        self.text.setObjectName("text")
        root.addWidget(self.text, 1)

        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.copy_btn = QPushButton("\U0001F4CB  Copiar")
        self.copy_btn.clicked.connect(self.copy_text)
        self.clear_btn = QPushButton("\U0001F5D1  Limpar")
        self.clear_btn.clicked.connect(self.text.clear)
        self.quit_btn = QPushButton("Sair")
        self.quit_btn.setObjectName("quit")
        self.quit_btn.clicked.connect(self.quit_app)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.clear_btn)
        btns.addStretch()
        btns.addWidget(self.quit_btn)
        root.addLayout(btns)

        self.setStyleSheet(STYLE)

        # espera o servidor ficar pronto
        self.ready_thread = ReadyThread()
        self.ready_thread.ready.connect(self.on_ready)
        self.ready_thread.start()

    # -- estado do servidor --
    def on_ready(self):
        self.mic.set_state(MicButton.IDLE)
        self.status.setText(f"Pronto — clique no microfone e fale  ({self.channels}ch → c{MIC_CHANNEL})")

    # -- gravacao --
    def toggle_record(self):
        if self.mic.state == MicButton.IDLE:
            self.start_record()
        elif self.mic.state == MicButton.RECORDING:
            self.stop_record()

    def start_record(self):
        self.server.ensure_running()  # ressuscita o motor se ele tiver caido
        fd, self.raw_path = tempfile.mkstemp(prefix="voz-", suffix=".pcm")
        os.close(fd)
        try:
            # pw-record nativo em RAW (PCM cru) p/ ler o audio enquanto grava.
            # parecord (Pulse) entrega zeros com a Scarlett 18i20 gen1 pro-audio.
            cmd = ["pw-record", "--raw"]
            if self.target:
                cmd += ["--target", self.target]   # senao: mic padrao do sistema
            cmd += ["--channels", str(self.channels), "--rate", str(REC_RATE),
                    "--format", "s16", self.raw_path]
            self.rec_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.status.setText(f"Erro ao gravar: {e}")
            return
        self._base = self.text.toPlainText()
        self._stopping = False
        self._partial_busy = False
        self._retried = False
        self.mic.set_state(MicButton.RECORDING)
        self.status.setText("●  Ouvindo… (ao vivo) — clique para parar")
        self.partial_timer.start()

    def on_partial_tick(self):
        """A cada ~1.7s, retranscreve o que ja foi falado e atualiza a tela."""
        if self._partial_busy or self._stopping or not self.raw_path:
            return
        try:
            size = os.path.getsize(self.raw_path)
        except OSError:
            return
        if size < int(REC_RATE * self.channels * 2 * 0.6):  # ~0.6s de audio
            return
        self._partial_busy = True
        self.partial_worker = Worker(self.raw_path, self.channels, final=False)
        self.partial_worker.done.connect(self.on_result)
        self.partial_worker.failed.connect(self.on_worker_fail)
        self.partial_worker.start()

    def stop_record(self):
        self._stopping = True
        self.partial_timer.stop()
        if self.rec_proc and self.rec_proc.poll() is None:
            self.rec_proc.terminate()
            try:
                self.rec_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.rec_proc.kill()
        self.rec_proc = None
        self.mic.set_state(MicButton.BUSY)
        self.status.setText("Finalizando…")
        self.final_worker = Worker(self.raw_path, self.channels, final=True)
        self.final_worker.done.connect(self.on_result)
        self.final_worker.failed.connect(self.on_worker_fail)
        self.final_worker.start()

    def _compose(self, addition):
        base = self._base
        sep = "" if (not base or base.endswith((" ", "\n"))) else " "
        return base + sep + addition

    def _show_text(self, full):
        self.text.setPlainText(full)
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text.setTextCursor(cursor)

    def on_result(self, txt, final):
        if not final:
            self._partial_busy = False
            if self._stopping:
                return  # a passada final assume o controle
            if txt:
                self._show_text(self._compose(txt))
            return
        # passada final (no 'parar')
        self._retried = False
        full = self._compose(txt) if txt else self._base
        self._show_text(full)
        self._base = full
        self.mic.set_state(MicButton.IDLE)
        self.status.setText("Pronto — clique no microfone e fale"
                            if txt else "Nao entendi nada — tente de novo")
        self._cleanup_raw()

    def on_worker_fail(self, msg, final):
        if not final:
            self._partial_busy = False
            return
        # erro de conexao = motor caiu; ressuscita e refaz a passada 1x
        conn_err = any(k in msg for k in
                       ("Connection refused", "Failed to establish",
                        "Max retries", "Connection aborted"))
        if conn_err and not self._retried and self.raw_path:
            self._retried = True
            self.mic.set_state(MicButton.BUSY)
            self.status.setText("Motor caiu — reiniciando e refazendo…")
            self.server.ensure_running()
            self.retry_thread = ReadyThread()
            self.retry_thread.ready.connect(self._retry_final)
            self.retry_thread.start()
            return
        self._retried = False
        self.mic.set_state(MicButton.IDLE)
        self.status.setText(msg)
        self._cleanup_raw()

    def _retry_final(self):
        """Refaz a passada final depois que o motor voltou a ficar pronto."""
        if not self.raw_path:
            self.mic.set_state(MicButton.IDLE)
            return
        self.final_worker = Worker(self.raw_path, self.channels, final=True)
        self.final_worker.done.connect(self.on_result)
        self.final_worker.failed.connect(self.on_worker_fail)
        self.final_worker.start()

    def _cleanup_raw(self):
        if self.raw_path:
            try:
                os.remove(self.raw_path)
            except OSError:
                pass
            self.raw_path = None

    # -- botoes --
    def copy_text(self):
        QGuiApplication.clipboard().setText(self.text.toPlainText())
        self.status.setText("Texto copiado ✓")

    # -- ciclo de vida --
    def show_raise(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, ev):
        if self._really_quit:
            ev.accept()
        else:
            ev.ignore()
            self.hide()  # mantem o modelo quente

    def quit_app(self):
        self._really_quit = True
        self.server.stop()
        QApplication.quit()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.hide()
        elif ev.key() == Qt.Key_Space and self.mic.isEnabled():
            self.toggle_record()
        else:
            super().keyPressEvent(ev)


# ----------------------------- visual --------------------------------------
STYLE = """
QWidget { background: #16161c; color: #e9e9f0;
          font-family: 'Inter','Noto Sans',sans-serif; font-size: 14px; }
QLabel#title { font-size: 26px; font-weight: 800; color: #ffffff;
               letter-spacing: 6px; padding-top: 4px; }
QLabel#subtitle { color: #6c6c80; font-size: 10px; letter-spacing: 4px; }
QLabel#status { color: #9a9ab0; font-size: 13px; }
QTextEdit#text { background: #20202a; border: 1px solid #2e2e3a;
                 border-radius: 12px; padding: 12px; font-size: 16px;
                 line-height: 1.5; selection-background-color: #7c5cff; }
QPushButton { background: #2a2a36; border: none; border-radius: 10px;
              padding: 11px 18px; font-size: 14px; color: #e9e9f0; }
QPushButton:hover { background: #34344a; }
QPushButton:pressed { background: #3d3d57; }
QPushButton#quit { background: transparent; color: #8a8a9a; }
QPushButton#quit:hover { background: #2a2a36; color: #ff6b7a; }
MicButton { background: transparent; }
"""


def make_icon():
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor("#7c5cff")))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setBrush(QBrush(QColor("white")))
    p.drawRoundedRect(QRectF(26, 16, 12, 22), 6, 6)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor("white"), 3))
    p.drawArc(QRectF(20, 22, 24, 24), 200 * 16, 140 * 16)
    p.drawLine(32, 40, 32, 48)
    p.end()
    return QIcon(pm)


# ----------------------------- main ----------------------------------------
def main():
    # se ja houver uma instancia, apenas pede para ela aparecer
    probe = QLocalSocket()
    probe.connectToServer(SOCKET_NAME)
    if probe.waitForConnected(250):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)

    server = WhisperServer()
    server.start()

    win = MainWindow(server)

    # servidor local p/ instancia unica (Meta+H -> mostra a janela existente)
    QLocalServer.removeServer(SOCKET_NAME)
    local = QLocalServer()
    local.listen(SOCKET_NAME)

    def on_conn():
        c = local.nextPendingConnection()
        c.readyRead.connect(lambda: (c.readAll(), win.show_raise()))

    local.newConnection.connect(on_conn)

    # bandeja do sistema (opcional)
    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(make_icon(), app)
        tray.setToolTip(APP_NAME)
        menu = QMenu()
        act_show = QAction("Mostrar", app)
        act_show.triggered.connect(win.show_raise)
        act_quit = QAction("Sair", app)
        act_quit.triggered.connect(win.quit_app)
        menu.addAction(act_show)
        menu.addAction(act_quit)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda r: win.show_raise()
            if r == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray.show()

    app.aboutToQuit.connect(server.stop)
    win.show_raise()
    return app.exec()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(main())

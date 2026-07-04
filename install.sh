#!/usr/bin/env bash
# NEUROVOX — instalador. Compila o motor (whisper.cpp), baixa o modelo,
# cria o ambiente Python e instala o atalho de aplicativo.
#
#   ./install.sh          # motor CPU (estavel em qualquer maquina)
#   ./install.sh --gpu    # tambem compila o motor Vulkan/GPU (mais rapido)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${NEUROVOX_MODEL:-large-v3-turbo}"
WANT_GPU=0
[ "${1:-}" = "--gpu" ] && WANT_GPU=1

say() { printf '\n\033[1;35m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERRO:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. dependencias ---
say "Checando dependencias"
miss=()
for c in git cmake make gcc g++ ffmpeg python3 pw-record; do
  command -v "$c" >/dev/null 2>&1 || miss+=("$c")
done
if [ ${#miss[@]} -gt 0 ]; then
  echo "Faltando: ${miss[*]}"
  echo "  Fedora:  sudo dnf install git cmake gcc gcc-c++ make ffmpeg-free pipewire-utils python3"
  echo "  Debian/Ubuntu:  sudo apt install git cmake build-essential ffmpeg pipewire-bin python3-venv"
  echo "  Arch:  sudo pacman -S git cmake base-devel ffmpeg pipewire python"
  die "instale os pacotes acima e rode de novo."
fi

# --- 2. whisper.cpp ---
if [ ! -d "$DIR/whisper.cpp/.git" ]; then
  say "Clonando whisper.cpp"
  git clone --depth=1 https://github.com/ggml-org/whisper.cpp.git "$DIR/whisper.cpp"
fi

# --- 3. compilar motor CPU ---
say "Compilando motor CPU (build-cpu)"
cmake -S "$DIR/whisper.cpp" -B "$DIR/whisper.cpp/build-cpu" \
      -DGGML_VULKAN=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$DIR/whisper.cpp/build-cpu" --target whisper-server whisper-cli \
      -j"$(nproc)" >/dev/null
echo "  ok: whisper.cpp/build-cpu/bin/whisper-server"

# --- 3b. motor GPU (opcional) ---
if [ "$WANT_GPU" = 1 ]; then
  say "Compilando motor GPU/Vulkan (build) — requer vulkan-headers, glslc, spirv-tools"
  cmake -S "$DIR/whisper.cpp" -B "$DIR/whisper.cpp/build" \
        -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release >/dev/null || \
        die "falha ao configurar Vulkan (instale vulkan-loader-devel vulkan-headers glslc spirv-tools-devel glslang-devel)"
  cmake --build "$DIR/whisper.cpp/build" --target whisper-server -j"$(nproc)" >/dev/null
  echo "  ok: whisper.cpp/build/bin/whisper-server  (troque \"engine\":\"gpu\" no config.json)"
fi

# --- 4. modelo ---
if [ ! -f "$DIR/whisper.cpp/models/ggml-$MODEL.bin" ]; then
  say "Baixando modelo: $MODEL (pode ser ~1.6 GB)"
  ( cd "$DIR/whisper.cpp" && bash models/download-ggml-model.sh "$MODEL" )
fi

# --- 5. ambiente Python ---
say "Criando venv e instalando dependencias Python"
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --quiet --upgrade pip
"$DIR/venv/bin/pip" install --quiet -r "$DIR/requirements.txt"

# --- 6. config local ---
if [ ! -f "$DIR/config.json" ]; then
  cp "$DIR/config.example.json" "$DIR/config.json"
  echo "  criado config.json (edite p/ mic/idioma/motor)"
fi

# --- 7. atalho de aplicativo ---
say "Instalando lancador"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/neurovox.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NEUROVOX
Comment=Ditado por voz offline (fala -> texto)
Exec=$DIR/voz.sh
Icon=audio-input-microphone
Terminal=false
Categories=Utility;AudioVideo;
StartupNotify=false
StartupWMClass=voz_app.py
EOF
update-desktop-database "$APPS" 2>/dev/null || true

say "Pronto! ✅"
cat <<EOF

  Rodar agora:   $DIR/voz.sh
  Config:        $DIR/config.json

  Atalho global (KDE):  Configuracoes > Atalhos > Adicionar > Comando
                        Comando: $DIR/voz.sh   |   Tecla: Meta+H
  (GNOME: Configuracoes > Teclado > Atalhos personalizados)

EOF

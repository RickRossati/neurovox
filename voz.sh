#!/usr/bin/env bash
# Launcher do "Voz -> Texto". Abrir de novo apenas foca a janela existente.
cd "$HOME/voz" || exit 1
exec "$HOME/voz/venv/bin/python" "$HOME/voz/voz_app.py" "$@"

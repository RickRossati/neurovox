# NEUROVOX

**Ditado por voz offline para Linux** — fala no microfone, vira texto. Local, privado,
sem nuvem, rodando na sua própria máquina com [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

> Aperte um atalho, clique no microfone, fale — o texto aparece **ao vivo** e você copia.
> Pensado para PT-BR, mas funciona em qualquer idioma que o Whisper suporta.

---

## ✨ Recursos

- 🎙️ **Ditado por voz → texto**, 100% **offline** e privado
- ⚡ **Transcrição ao vivo** — o texto vai surgindo enquanto você fala
- 🧠 Motor **whisper.cpp** com modelo `large-v3-turbo` (ótima qualidade em PT-BR)
- 🖥️ Roda em **CPU** (estável em qualquer máquina) ou **GPU/Vulkan** (opcional, mais rápido)
- 🔥 Modelo fica **quente na memória** (whisper-server) → cada ditado é rápido
- 🪟 Interface Qt limpa, com atalho global (ex.: `Win+H`), botões **Copiar** / **Limpar**
- 🔧 **Configurável** por `config.json` (microfone, idioma, modelo, motor)

## 📦 Requisitos

- Linux com **PipeWire** (KDE/GNOME/etc.)
- `python3` (3.10+), `git`, `cmake`, `gcc/g++`, `make`, `ffmpeg`, `pw-record` (pipewire-utils)
- ~2 GB de disco para o modelo

## 🚀 Instalação

```bash
git clone https://github.com/RickRossati/neurovox.git
cd neurovox
./install.sh            # motor CPU (recomendado)
# ./install.sh --gpu    # também compila o motor Vulkan/GPU (opcional)
```

O `install.sh` compila o motor, baixa o modelo (~1.6 GB), cria o ambiente Python
e instala o lançador. No fim ele mostra como registrar o atalho global.

## ▶️ Uso

```bash
./voz.sh
```

1. Registre o atalho global (o instalador mostra como) e abra com **`Win+H`**
2. Clique no **microfone** → fale → clique de novo para parar
3. O texto aparece **ao vivo**; use **Copiar** para levar para qualquer lugar

## ⚙️ Configuração — `config.json`

| Campo | Padrão | Descrição |
|---|---|---|
| `language` | `pt` | idioma do reconhecimento (`auto` para detectar) |
| `model` | `large-v3-turbo` | modelo da **passada final** (qualidade) |
| `model_live` | `small` | modelo leve das **parciais ao vivo** (velocidade); `base` = ainda mais rápido; igual a `model` = desliga o híbrido |
| `engine` | `cpu` | `cpu` (estável) ou `gpu` (Vulkan, mais rápido) |
| `device` | `default` | mic padrão do sistema, ou o nome de uma fonte do PipeWire |
| `channels` / `channel` | `1` / `0` | para interfaces multicanal: grava N canais e extrai um |
| `threads` | `8` | threads do motor CPU |

**Interface de áudio multicanal?** (ex.: Focusrite Scarlett) — aponte `device` para o
nome da fonte (`pactl list short sources`), defina `channels` para o total e `channel`
para o índice do seu mic (0 = primeiro).

## 🖥️ CPU vs GPU

- **CPU** (padrão): compila `whisper.cpp` sem Vulkan; roda em qualquer lugar. Num CPU
  moderno o `large-v3-turbo` transcreve bem mais rápido que tempo real.
- **GPU/Vulkan** (`--gpu`): mais rápido, mas o backend Vulkan do ggml pode **falhar em
  alguns drivers/GPUs** (ex.: NVIDIA Blackwell com certos drivers dá segfault na init).
  Se a GPU falhar, é só manter `"engine": "cpu"`.

## 🧩 Como funciona

```
mic (PipeWire, pw-record) → ffmpeg (extrai canal + normaliza) →
whisper-server (modelo quente) → texto ao vivo na UI
```

- Captura com **`pw-record` nativo** (a camada de compat do PulseAudio pode entregar
  silêncio em algumas interfaces — PipeWire nativo resolve).
- O modelo fica carregado num `whisper-server` local; a UI só manda o áudio e recebe texto.
- Transcrição ao vivo = **dois motores**: um modelo leve (`small`) transcreve o buffer
  a cada ~1.7 s pras parciais aparecerem rápido, e o modelo principal (`large-v3-turbo`)
  faz a passada final caprichada quando você para.

## 🗺️ Roadmap

- [ ] Colar direto no campo focado (além de copiar)
- [ ] Seletor de microfone e modelo na UI
- [ ] Pontuação/formatização automática
- [ ] Empacotamento (AppImage / Flatpak)

## 🙏 Créditos

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (ggml-org)
- [Whisper](https://github.com/openai/whisper) (OpenAI)

## 📄 Licença

[MIT](LICENSE) © Ricardo Rossati (Mister RickRoss)

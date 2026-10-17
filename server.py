import http.server
import os
import queue
import random
import socketserver
import struct
import threading
import numpy as np

# --- KONFIGURACJA AUDIO ---
SAMPLE_RATE = 44100
TEMPO = 114
BEAT = 60.0 / TEMPO
BAR = BEAT * 4

# Render automatycznie przypisuje port przez zmienną środowiskową (lokalnie fallback na 10000)
PORT = int(os.environ.get("PORT", 10000))


def midi_to_freq(midi_note):
  return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


# --- CZYSTY I MIĘKKI SYNTEZATOR (Bez dysonansów) ---
def synth_note(freq, duration, instrument="chord", velocity=0.8):
  num_samples = int(SAMPLE_RATE * duration)
  if num_samples <= 0:
    return np.zeros(0)
  t = np.linspace(0, duration, num_samples, endpoint=False)

  if instrument == "bass":
    wave = (
        np.sin(2 * np.pi * freq * t) * 0.8
        + np.sin(2 * np.pi * freq * 2 * t) * 0.2
    )
    envelope = np.exp(-3.5 * t) * (1 - np.exp(-70.0 * t))
  elif instrument == "chord":
    wave = (
        np.sin(2 * np.pi * freq * t) * 0.6
        + np.sin(2 * np.pi * freq * 2 * t) * 0.2
    )
    envelope = np.exp(-2.0 * t) * (1 - np.exp(-30.0 * t))
  elif instrument == "lead":
    vibrato = 4.0 * np.sin(2 * np.pi * 5.0 * t)
    freq_mod = freq + vibrato
    wave = (
        np.sin(2 * np.pi * freq_mod * t) * 0.7
        + np.sin(2 * np.pi * freq_mod * 2 * t) * 0.15
    )
    envelope = np.exp(-1.2 * t) * (1 - np.exp(-25.0 * t))
  elif instrument == "drum_kick":
    freq_env = 80 * np.exp(-22 * t) + 35
    wave = np.sin(2 * np.pi * freq_env * t)
    envelope = np.exp(-9 * t)
  elif instrument == "drum_rim":
    wave = np.random.uniform(-1, 1, num_samples)
    envelope = np.exp(-50 * t)
  elif instrument == "drum_shaker":
    wave = np.random.uniform(-1, 1, num_samples)
    envelope = np.exp(-45 * t) * (t < 0.08)
  else:
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-3 * t)

  return wave * envelope * velocity


# --- HARMONIJNY GENERATOR TAKTU ---
def create_bar_audio(root_midi, chord_notes, scale_notes):
  bar_samples = int(SAMPLE_RATE * BAR)
  buffer = np.zeros(bar_samples)

  def add_sound(audio_piece, start_time):
    idx = int(start_time * SAMPLE_RATE)
    end_idx = idx + len(audio_piece)
    if end_idx <= bar_samples:
      buffer[idx:end_idx] += audio_piece[: bar_samples - idx]
    elif idx < bar_samples:
      buffer[idx:] += audio_piece[: bar_samples - idx]

  eighth = BEAT / 2
  for i in range(8):
    vel = 0.22 if (i % 2 == 1) else 0.1
    add_sound(synth_note(0, eighth, "drum_shaker", velocity=vel), i * eighth)

  for tb in [BEAT * 1.5, BEAT * 3.5]:
    add_sound(synth_note(0, 0.12, "drum_rim", velocity=0.45), tb)

  for tb in [0.0, BEAT * 2.0]:
    add_sound(synth_note(0, 0.25, "drum_kick", velocity=0.6), tb)

  r_bass = root_midi - 12
  f_bass = r_bass + 7
  bass_hits = [
      (0.0, r_bass, 0.65),
      (BEAT * 1.5, f_bass, 0.45),
      (BEAT * 2.0, r_bass, 0.6),
      (BEAT * 3.5, f_bass, 0.45),
  ]
  for tb, m_note, vel in bass_hits:
    add_sound(
        synth_note(midi_to_freq(m_note), BEAT * 0.65, "bass", velocity=vel), tb
    )

  comp_times = [BEAT * 0.5, BEAT * 2.0, BEAT * 3.25]
  for tb in comp_times:
    for m_note in chord_notes:
      add_sound(
          synth_note(
              midi_to_freq(m_note), BEAT * 0.45, "chord", velocity=0.28
          ),
          tb,
      )

  if scale_notes:
    lead_scale = [n + 12 for n in scale_notes if 60 <= n + 12 <= 82]
    if lead_scale:
      current_note = random.choice(lead_scale)
      phrase_starts = [BEAT * 0.5, BEAT * 2.0]
      for tb in phrase_starts:
        step = random.choice([-1, 0, 1])
        try:
          idx = lead_scale.index(current_note)
          current_note = lead_scale[max(0, min(len(lead_scale) - 1, idx + step))]
        except ValueError:
          current_note = lead_scale[0]

        dur = random.choice([BEAT * 0.75, BEAT * 1.0])
        add_sound(
            synth_note(midi_to_freq(current_note), dur, "lead", velocity=0.45),
            tb,
        )

  return buffer


CHORD_PROGRESSION = [
    ("Fmaj7", 53, [53, 60, 64, 67], [53, 55, 57, 60, 62, 65]),
    ("Fmaj7", 53, [53, 60, 64, 67], [53, 55, 57, 60, 62, 65]),
    ("G9", 55, [55, 62, 65, 69], [55, 57, 59, 62, 64, 67]),
    ("G9", 55, [55, 62, 65, 69], [55, 57, 59, 62, 64, 67]),
    ("Gm7", 58, [58, 65, 69, 72], [58, 60, 62, 65, 67, 70]),
    ("C7", 60, [60, 67, 70, 74], [60, 62, 65, 67, 69, 72]),
    ("Fmaj7", 53, [53, 60, 64, 67], [53, 55, 57, 60, 62, 65]),
    ("A7", 57, [57, 64, 67, 73], [57, 60, 63, 65, 68, 70]),
]

global_audio_queue = queue.Queue(maxsize=10)


def music_producer_loop():
  chorus_num = 1
  while True:
    chorus_buffers = []
    for chord_name, root, notes, scale in CHORD_PROGRESSION:
      bar_audio = create_bar_audio(root, notes, scale)
      chorus_buffers.append(bar_audio)

    full_chorus = np.concatenate(chorus_buffers)

    delay_samples = int(0.25 * SAMPLE_RATE)
    if len(full_chorus) > delay_samples:
      full_chorus[delay_samples:] += full_chorus[:-delay_samples] * 0.15

    max_val = np.max(np.abs(full_chorus))
    if max_val > 0:
      full_chorus = full_chorus / max_val * 0.7

    audio_int16 = (full_chorus * 32767).astype(np.int16)
    stereo_pcm = np.column_stack((audio_int16, audio_int16)).flatten()

    global_audio_queue.put((chorus_num, stereo_pcm.tobytes()))
    chorus_num += 1


class RadioStreamHandler(http.server.BaseHTTPRequestHandler):

  def do_GET(self):
    if self.path == "/" or self.path == "/stream.wav":
      self.send_response(200)
      self.send_header("Content-Type", "audio/wav")
      self.send_header("Transfer-Encoding", "chunked")
      self.end_headers()

      channels = 2
      bits_per_sample = 16
      byte_rate = SAMPLE_RATE * channels * (bits_per_sample // 8)
      block_align = channels * (bits_per_sample // 8)

      wav_header = (
          b"RIFF"
          + struct.pack("<I", 0xFFFFFFFF)
          + b"WAVE"
          + b"fmt "
          + struct.pack("<I", 16)
          + struct.pack("<H", 1)
          + struct.pack("<H", channels)
          + struct.pack("<I", SAMPLE_RATE)
          + struct.pack("<I", byte_rate)
          + struct.pack("<H", block_align)
          + struct.pack("<H", bits_per_sample)
          + b"data"
          + struct.pack("<I", 0xFFFFFFFF)
      )

      try:
        self.wfile.write(wav_header)
        self.wfile.flush()

        while True:
          chorus_num, chunk_bytes = global_audio_queue.get()
          self.wfile.write(chunk_bytes)
          self.wfile.flush()
      except (ConnectionResetError, BrokenPipeError):
        pass
    else:
      self.send_response(404)
      self.end_headers()
      # POPRAWIONO: Zakodowanie polskich znaków do UTF-8 za pomocą .encode()
      self.wfile.write(
          "Strona nie znaleziona. Słuchaj radia pod adresem / lub /stream.wav".encode(
              "utf-8"
          )
      )

  def log_message(self, format, *args):
    pass


if __name__ == "__main__":
  producer_thread = threading.Thread(target=music_producer_loop, daemon=True)
  producer_thread.start()

  print(f"Uruchomiono radio na porcie {PORT}")

  with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), RadioStreamHandler) as httpd:
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      pass

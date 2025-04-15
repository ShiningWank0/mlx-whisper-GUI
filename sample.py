import pyaudio
import numpy as np
import mlx_whisper

# 音声キャプチャの設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024*2  # 5秒間の音声をキャプチャ

# PyAudioの初期化
audio = pyaudio.PyAudio()

# ストリームの開始
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

try:
    buffer=[]
    while True:
        # 音声データを取得
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(data, dtype=np.int16)/32768.0
            buffer.append(data)

            if len(buffer)>30:
                data = np.array(buffer)
                # NumPy配列に変換
                audio_data = data.reshape(data.shape[0]*data.shape[1])
                print("認識中")

                # 音声データを転写
                result = mlx_whisper.transcribe(audio_data,language="ja",
                                                path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
                print(result['text'])
                buffer=[]
                print("音声データ収集中")
        except Exception as e:
            print(e)
            pass



except KeyboardInterrupt:
    print("音声認識を終了します。")

finally:
    # ストリームの停止と終了
    stream.stop_stream()
    stream.close()
    audio.terminate()

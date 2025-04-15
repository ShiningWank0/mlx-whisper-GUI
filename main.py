import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue # スレッド間のデータ受け渡し用
import sounddevice as sd
import soundfile as sf
import numpy as np
import mlx_whisper
import datetime
import time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ––– グローバル変数 –––
selected_model = "tiny" # デフォルトモデル
# 利用可能なモデルリスト
available_models = ["tiny", "base", "small", "medium", "large"]
audio_stream = None
recording_thread = None
is_recording = False
output_file_realtime = None # リアルタイム文字起こし用ファイルオブジェクト
samplerate = 16000 # Whisperが期待するサンプルレート
channels = 1 # モノラル
blocksize = 1600 # 録音バッファサイズ (0.1秒分程度、調整可能)
audio_queue = queue.Queue() # 音声データキュー

# ––– 位置情報取得関数 –––
def get_location():
    """デバイスのおおよその現在地（緯度・経度）を取得する"""
    try:
        # user_agentは自身のアプリ名などに設定
        geolocator = Nominatim(user_agent="mlx_transcription_app_v1", timeout=10)
        # IPアドレスからおおよその位置を推定 (精度は限定的)
        # 注意: ネットワーク接続が必要、利用規約を確認
        location = geolocator.geocode("my location") # IPベースの位置情報取得を試みる

        if location:
            print(f"取得した位置情報: {location.address} ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            # 現在地の情報を直接取得するのは難しい場合があるため、固定値を返す例
            print("IPベースの位置情報取得に失敗。デフォルト値（名古屋駅付近）を返します。")
            # messagebox.showwarning("位置情報エラー", "IPアドレスからの位置情報取得に失敗しました。\nデフォルト値（名古屋）を使用します。")
            return 35.1709, 136.8815 # 名古屋駅付近の緯度経度 (フォールバック)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"位置情報取得サービスエラー: {e}")
        messagebox.showerror("位置情報エラー", f"位置情報サービスの利用中にエラーが発生しました。\nネットワーク接続を確認してください。\n{e}")
        return "エラー", "エラー"
    except Exception as e:
        # requests.exceptions.RequestException などの可能性
        print(f"位置情報取得中に予期せぬエラー: {e}")
        messagebox.showerror("位置情報エラー", f"位置情報取得中に予期せぬエラーが発生しました。\n{e}")
        return "エラー", "エラー"

# ––– 音声コールバック関数(リアルタイム録音用) –––
def audio_callback(indata, frames, time, status):
    """マイクから音声データを受け取りキューに入れる"""
    if status:
        print(f"オーディオコールバックステータス: {status}", flush=True)
    # 受け取ったデータをそのままキューに入れる
    audio_queue.put(indata.copy())

# ––– リアルタイム文字起こし処理関数(別スレッドで実行) –––
def transcribe_realtime_task(model_name, output_filename, app_instance):
    global is_recording, output_file_realtime, samplerate, audio_queue
    
    try:
        # 位置情報取得
        latitude, longitude = get_location()
        if latitude == "エラー":
            location_info = "録音開始地点: 位置情報取得エラー\n---\n"
        else:
            location_info = f"録音開始地点: 緯度 {latitude:.6f}, 経度 {longitude:.6f}\n---\n"
        print(f"リアルタイム録音 - {location_info.strip()}")
        # GUIにも反映(メインスレッドで実行させる)
        app_instance.after(0, lambda: app_instance.update_transcription_display(location_info))

        # モデルロード
        print(f"モデル '{model_name}' をロード中...")
        app_instance.after(0, lambda: app_instance.status_label.configure(text=f"モデル '{model_name}' をロード中..."))
        model = mlx_whisper.load_model(model_name, verbose=False)
        print("モデルのロード完了。")
        app_instance.after(0, lambda: app_instance.status_label.configure(text=f"リアルタイム文字起こし中 ({model_name})..."))

        # 出力ファイルを開く
        with open(output_filename, "w", encoding="utf-8") as f:
            output_file_realtime = f
            output_file_realtime.write(location_info) # ファイルの先頭に位置情報を書き込む
            print(f"出力ファイル '{output_filename}' を開きました。")
            accumulated_audio = np.array([], dtype=np.float32) # 音声データを蓄積するバッファ
            min_chunk_duration = 1.0 # 最低でも1秒分のデータを処理(調整可能)
            max_chunk_duration = 10.0 # 最大10秒分まで溜めて処理(調整可能)

            while is_recording:
                try:
                    # キューから音声データを取得(非ブロッキング)
                    while not audio_queue.empty():
                        audio_chunk = audio_queue.get_nowait()
                        accumulated_audio = np.concatenate((accumulated_audio, audio_chunk.flatten()))
                    
                    # 蓄積された音声データの長さを計測
                    duration = len(accumulated_audio) / samplerate

                    # 一定時間分のデータが溜まったか、停止指示が出たら処理
                    # is_recordingがFalseになった場合も残りを処理
                    if duration >= min_chunk_duration or (not is_recording and len(accumulated_audio) > 0):
                        # 処理する音声データを選択 (最大長を超えないように)
                        process_len = min(len(accumulated_audio), int(max_chunk_duration * samplerate))
                        audio_to_process = accumulated_audio[:process_len]
                        # 残りのデータをバッファに残す
                        accumulated_audio = accumulated_audio[process_len:]

                        if len(audio_to_process) > 0:
                            print(f"音声チャンク (長さ: {len(audio_to_process)/samplerate:.2f}秒) を処理中...")
                            # 文字起こし実行 (リアルタイムでは単語タイムスタンプはオフ推奨)
                            result = mlx_whisper.transcribe(model, audio_to_process,
                                                            sample_rate=samplerate,
                                                            language="ja", # 日本語を指定
                                                            word_timestamps=False,
                                                            verbose=False) # verbose=Trueでログ増加
                            transcript = result["text"].strip()
                            print(f"文字起こし結果: '{transcript}'")

                            if transcript: # 文字起こし結果が空でなければ
                                current_time_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # 日本時間 (ミリ秒まで)
                                line_to_write = f"[{current_time_jst}] {transcript}\n"
                                output_file_realtime.write(line_to_write)
                                output_file_realtime.flush() # バッファをフラッシュして即時書き込み
                                print(f"ファイルに書き込み: {line_to_write.strip()}")
                                # GUIのテキストエリアにも表示 (メインスレッドで実行)
                                app_instance.after(0, lambda text=line_to_write: app_instance.update_transcription_display(text))
                        # is_recordingがFalseになったらループを抜ける前に残りを処理したことになる
                        if not is_recording and len(accumulated_audio) == 0:
                            break
                    else:
                        # データが溜まっていなければ少し待つ
                        time.sleep(0.1) # CPU負荷を軽減

                except queue.Empty:
                    # キューが空の場合（かつ録音中の場合）は少し待つ
                    if is_recording:
                        time.sleep(0.1)
                    elif len(accumulated_audio) > 0:
                        # 録音停止後、まだバッファにデータがあれば最後の処理へ
                        continue
                    else:
                        # 録音停止後、バッファも空ならループを抜ける
                        break
                except Exception as e:
                    print(f"リアルタイム文字起こし処理中にエラー発生: {e}")
                    # エラー内容をGUIに通知することも検討
                    app_instance.after(0, lambda: messagebox.showerror("処理エラー", f"文字起こし処理中にエラーが発生しました:\n{e}"))
                    # 継続困難なエラーの場合はループを抜ける
                    break

    except sd.PortAudioError as e:
        print(f"オーディオデバイスエラー: {e}")
        app_instance.after(0, lambda: messagebox.showerror("オーディオエラー", f"マイクの初期化またはストリーム中にエラーが発生しました。\n{e}"))
        # エラー発生時は録音状態をリセットする必要がある
        app_instance.after(0, app_instance.force_stop_realtime) # GUIの状態もリセット
    except RuntimeError as e:
        print(f"ランタイムエラー（モデル関連の可能性）: {e}")
        app_instance.after(0, lambda: messagebox.showerror("実行時エラー", f"処理中に実行時エラーが発生しました。\nモデルデータやメモリを確認してください。\n{e}"))
        app_instance.after(0, app_instance.force_stop_realtime)
    except Exception as e:
        print(f"リアルタイム文字起こしタスクで予期せぬエラー: {e}")
        import traceback
        traceback.print_exc() # 詳細なトレースバックを出力
        app_instance.after(0, lambda: messagebox.showerror("予期せぬエラー", f"リアルタイム文字起こし中に予期せぬエラーが発生しました:\n{e}"))
        app_instance.after(0, app_instance.force_stop_realtime)
    finally:
        if output_file_realtime and not output_file_realtime.closed:
             try:
                output_file_realtime.close()
                print("出力ファイル（リアルタイム）を閉じました。")
             except Exception as e:
                print(f"出力ファイル（リアルタイム）クローズエラー: {e}")
        output_file_realtime = None # グローバル変数をクリア
        # is_recording は stop_realtime_transcription で False になる想定だが、念のため
        # is_recording = False
        print("リアルタイム文字起こしスレッド終了。")
        app_instance.after(0, lambda: app_instance.status_label.configure(text="リアルタイム文字起こし停止"))


# --- GUI アプリケーションクラス ---
class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("文字起こしアプリ (mlx-whisper)")
        self.geometry("700x600") # 少し広めに

        # Appearance and Theme
        ctk.set_appearance_mode("System") # System, Light, Dark
        ctk.set_default_color_theme("blue") # Themes: blue (default), dark-blue, green

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # --- Top Control Frame ---
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(pady=10, padx=10, fill="x")

        # Model Selection
        self.model_label = ctk.CTkLabel(self.control_frame, text="モデル:")
        self.model_label.pack(side="left", padx=(0, 5))
        self.model_var = ctk.StringVar(value=selected_model)
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=available_models, command=self.on_model_select)
        self.model_dropdown.pack(side="left", padx=5)

        # Status Label
        self.status_label = ctk.CTkLabel(self.control_frame, text="準備完了", width=200)
        self.status_label.pack(side="right", padx=(5, 0))


        # --- File Transcription Frame ---
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(pady=10, padx=10, fill="x")

        self.file_title_label = ctk.CTkLabel(self.file_frame, text="ファイルから文字起こし", font=ctk.CTkFont(weight="bold"))
        self.file_title_label.pack(anchor="w", pady=(0, 5))

        self.file_selection_frame = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        self.file_selection_frame.pack(fill="x")

        self.select_file_button = ctk.CTkButton(self.file_selection_frame, text="音声ファイル選択", command=self.select_file)
        self.select_file_button.pack(side="left", padx=(0, 5))

        self.filepath_label = ctk.CTkLabel(self.file_selection_frame, text="ファイルが選択されていません", wraplength=350, anchor="w")
        self.filepath_label.pack(side="left", padx=5, fill="x", expand=True)

        self.transcribe_file_button = ctk.CTkButton(self.file_frame, text="文字起こし開始 (ファイル)", command=self.start_file_transcription, state="disabled")
        self.transcribe_file_button.pack(anchor="e", pady=(5, 0))

        self.selected_filepath = None

        # --- Realtime Transcription Frame ---
        self.realtime_frame = ctk.CTkFrame(self.main_frame)
        self.realtime_frame.pack(pady=10, padx=10, fill="x")

        self.realtime_title_label = ctk.CTkLabel(self.realtime_frame, text="リアルタイム文字起こし (マイク)", font=ctk.CTkFont(weight="bold"))
        self.realtime_title_label.pack(anchor="w", pady=(0, 5))

        self.record_button = ctk.CTkButton(self.realtime_frame, text="録音開始", command=self.toggle_realtime_transcription, width=120)
        self.record_button.pack(anchor="w")


        # --- Transcription Display Area ---
        self.display_label = ctk.CTkLabel(self.main_frame, text="文字起こし結果:", font=ctk.CTkFont(weight="bold"))
        self.display_label.pack(anchor="w", padx=10, pady=(10, 0))

        self.transcription_textbox = ctk.CTkTextbox(self.main_frame, state="disabled", wrap="word") # wrap="word"で単語単位の折り返し
        self.transcription_textbox.pack(pady=(5, 10), padx=10, fill="both", expand=True)

    def on_model_select(self, choice):
        global selected_model
        selected_model = choice
        print(f"モデル '{selected_model}' が選択されました。")
        self.status_label.configure(text=f"モデル '{selected_model}' を選択")

    def select_file(self):
        # リアルタイム録音中はファイル選択不可
        if is_recording:
            messagebox.showwarning("録音中", "リアルタイム文字起こし中はファイルを選択できません。")
            return

        filepath = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if filepath:
            self.selected_filepath = filepath
            # ファイル名を短縮して表示（長すぎる場合）
            display_name = os.path.basename(filepath)
            if len(display_name) > 50:
                display_name = display_name[:20] + "..." + display_name[-27:]
            self.filepath_label.configure(text=display_name)
            self.transcribe_file_button.configure(state="normal")
            self.status_label.configure(text=f"ファイル選択: {os.path.basename(filepath)}")
            self.transcription_textbox.configure(state="normal")
            self.transcription_textbox.delete("1.0", tk.END)
            self.transcription_textbox.insert("1.0", f"ファイル '{os.path.basename(filepath)}' が選択されました。\n「文字起こし開始 (ファイル)」ボタンを押してください。")
            self.transcription_textbox.configure(state="disabled")
        else:
            # ファイル選択がキャンセルされた場合（何もしないか、状態をリセット）
            # self.selected_filepath = None
            # self.filepath_label.configure(text="ファイルが選択されていません")
            # self.transcribe_file_button.configure(state="disabled")
            # self.status_label.configure(text="ファイル選択がキャンセルされました")
            pass

    def set_ui_state(self, file_transcribing=False, realtime_recording=False):
        """UI要素の有効/無効を一括で設定するヘルパー関数"""
        if file_transcribing:
            self.model_dropdown.configure(state="disabled")
            self.select_file_button.configure(state="disabled")
            self.transcribe_file_button.configure(state="disabled")
            self.record_button.configure(state="disabled")
        elif realtime_recording:
            self.model_dropdown.configure(state="disabled")
            self.select_file_button.configure(state="disabled")
            self.transcribe_file_button.configure(state="disabled")
            self.record_button.configure(text="録音停止", state="normal") # 録音中は停止ボタンに
        else: # アイドル状態
            self.model_dropdown.configure(state="normal")
            self.select_file_button.configure(state="normal")
            # ファイルが選択されていればファイル文字起こしボタンを有効化
            self.transcribe_file_button.configure(state="normal" if self.selected_filepath else "disabled")
            self.record_button.configure(text="録音開始", state="normal")

    def start_file_transcription(self):
        if not self.selected_filepath:
            messagebox.showwarning("ファイル未選択", "文字起こしする音声ファイルを選択してください。")
            return

        model_name = self.model_var.get()
        input_filepath = self.selected_filepath

        # 出力ファイル名生成
        path_without_ext, _ = os.path.splitext(input_filepath)
        output_filepath = f"{path_without_ext}_transcribe.txt" # 同じディレクトリに保存

        self.status_label.configure(text=f"ファイル文字起こし準備中...")
        self.set_ui_state(file_transcribing=True) # UIを無効化
        self.transcription_textbox.configure(state="normal")
        self.transcription_textbox.delete("1.0", tk.END)
        self.transcription_textbox.insert("1.0", f"ファイル: {os.path.basename(input_filepath)}\nモデル: {model_name}\n文字起こしを開始します...\n\n")
        self.transcription_textbox.configure(state="disabled")
        self.update() # GUIを即時更新

        # 別スレッドで文字起こしを実行
        threading.Thread(target=self.transcribe_file_task, args=(model_name, input_filepath, output_filepath), daemon=True).start()

    def transcribe_file_task(self, model_name, input_filepath, output_filepath):
        """ファイル文字起こしを別スレッドで実行する"""
        try:
            # モデルロード
            self.after(0, lambda: self.status_label.configure(text=f"モデル '{model_name}' をロード中..."))
            print(f"ファイル文字起こし - モデル '{model_name}' をロード中...")
            model = mlx_whisper.load_model(model_name, verbose=False)
            print("ファイル文字起こし - モデルのロード完了。")
            self.after(0, lambda: self.status_label.configure(text=f"ファイル文字起こし中..."))

            # 文字起こし実行 (word_timestamps=True で単語/セグメントのタイムスタンプ取得)
            print(f"ファイル '{input_filepath}' の文字起こしを開始...")
            start_time = time.time()
            result = mlx_whisper.transcribe(model, input_filepath,
                                            language="ja", # 日本語指定
                                            word_timestamps=True, # タイムスタンプ有効化
                                            verbose=False) # Trueで詳細ログ
            end_time = time.time()
            print(f"ファイル文字起こし完了。処理時間: {end_time - start_time:.2f}秒")

            # 結果をファイルに書き込み & GUIに表示
            self.after(0, lambda: self.process_file_transcription_result(result, input_filepath, output_filepath, model_name))

        except FileNotFoundError:
             self.after(0, lambda: messagebox.showerror("エラー", f"ファイルが見つかりません: {input_filepath}"))
             self.after(0, lambda: self.status_label.configure(text="エラー: ファイル未検出"))
        except Exception as e:
            print(f"ファイル文字起こし中にエラー発生: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("エラー", f"ファイル文字起こし中にエラーが発生しました:\n{e}"))
            self.after(0, lambda: self.status_label.configure(text="エラー発生"))
        finally:
            # ボタンの状態を元に戻す (メインスレッドで実行)
            self.after(0, lambda: self.set_ui_state(file_transcribing=False))

    def process_file_transcription_result(self, result, input_filepath, output_filepath, model_name):
        """ファイル文字起こし結果を処理し、ファイル保存とGUI表示を行う"""
        try:
            output_content = []
            output_content.append(f"--- 文字起こし結果 ({model_name}) ---")
            output_content.append(f"ファイル: {os.path.basename(input_filepath)}")
            output_content.append(f"言語: {result.get('language', '不明')}")
            output_content.append("\n--- 全文 ---")
            output_content.append(result.get('text', '文字起こし結果がありません。'))
            output_content.append("\n--- 発言タイミング (セグメント) ---")

            if "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    start = segment.get('start')
                    end = segment.get('end')
                    text = segment.get('text', '').strip()
                    if start is not None and end is not None:
                        # 秒を HH:MM:SS.ms 形式に変換
                        start_time_str = str(datetime.timedelta(seconds=start)).split('.')[0] + '.' + f"{int(start % 1 * 1000):03d}"
                        end_time_str = str(datetime.timedelta(seconds=end)).split('.')[0] + '.' + f"{int(end % 1 * 1000):03d}"
                        output_content.append(f"[{start_time_str} --> {end_time_str}] {text}")
                    else:
                        output_content.append(f"[タイムスタンプ不明] {text}")
            else:
                output_content.append("タイムスタンプ情報（セグメント）は利用できませんでした。")

            # 単語ごとのタイムスタンプも必要であれば追加 (コメントアウト)
            # output_content.append("\n--- 発言タイミング (単語) ---")
            # word_timestamps_available = False
            # if "segments" in result and result["segments"]:
            #     for segment in result["segments"]:
            #          if "words" in segment and segment["words"]:
            #              word_timestamps_available = True
            #              for word_info in segment["words"]:
            #                  start = word_info.get('start')
            #                  end = word_info.get('end')
            #                  word = word_info.get('word', '')
            #                  if start is not None and end is not None:
            #                      start_time_str = str(datetime.timedelta(seconds=start)).split('.')[0] + '.' + f"{int(start % 1 * 1000):03d}"
            #                      end_time_str = str(datetime.timedelta(seconds=end)).split('.')[0] + '.' + f"{int(end % 1 * 1000):03d}"
            #                      output_content.append(f"[{start_time_str} --> {end_time_str}] {word}")
            #                  else:
            #                      output_content.append(f"[タイムスタンプ不明] {word}")
            # if not word_timestamps_available:
            #     output_content.append("タイムスタンプ情報（単語）は利用できませんでした。")

            # ファイルに書き込み
            full_output_text = "\n".join(output_content)
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(full_output_text)
            print(f"文字起こし結果を '{output_filepath}' に保存しました。")

            # GUIのテキストボックスに表示
            self.transcription_textbox.configure(state="normal")
            # self.transcription_textbox.delete("1.0", tk.END) # 既存の「開始します」メッセージを消去
            self.transcription_textbox.insert(tk.END, "\n" + full_output_text) # 結果を追記
            self.transcription_textbox.configure(state="disabled")
            self.status_label.configure(text=f"完了: {os.path.basename(output_filepath)}")
            messagebox.showinfo("完了", f"文字起こしが完了し、結果を\n{output_filepath}\nに保存しました。")

        except Exception as e:
            print(f"ファイル結果処理中にエラー: {e}")
            messagebox.showerror("結果処理エラー", f"文字起こし結果の処理中にエラーが発生しました:\n{e}")
            self.status_label.configure(text="結果処理エラー")


    def toggle_realtime_transcription(self):
        global is_recording
        if is_recording:
            self.stop_realtime_transcription()
        else:
            # ファイル文字起こし実行中は開始しない
            if self.transcribe_file_button.cget("state") == "disabled" and not is_recording:
                messagebox.showwarning("処理中", "ファイル文字起こし処理が完了するまでお待ちください。")
                return
            self.start_realtime_transcription()

    def start_realtime_transcription(self):
        global is_recording, audio_stream, recording_thread, audio_queue, samplerate, channels, blocksize

        if is_recording:
            return

        model_name = self.model_var.get()

        # 出力ファイル名生成 (日本時間)
        now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        output_filename = now_jst.strftime("%Y%m%d_%H%M%S") + "_realtime.txt"
        # 保存先ディレクトリ（例：アプリ実行ディレクトリ or ドキュメントフォルダなど）
        # output_dir = os.path.expanduser("~/Documents") # ドキュメントフォルダにする場合
        output_dir = os.getcwd() # 実行ディレクトリ
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

        # 状態変更とUI更新
        is_recording = True
        self.set_ui_state(realtime_recording=True) # UI状態を更新
        self.status_label.configure(text=f"リアルタイム文字起こし準備中...")
        self.transcription_textbox.configure(state="normal")
        self.transcription_textbox.delete("1.0", tk.END)
        self.transcription_textbox.insert("1.0", f"リアルタイム文字起こしを開始します...\nモデル: {model_name}\n出力ファイル: {output_filename}\n\n")
        self.transcription_textbox.configure(state="disabled")
        self.update()

        # キューをクリア (前回のデータが残らないように)
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            # オーディオストリーム開始
            print(f"オーディオストリームを開始します (Rate: {samplerate}, Channels: {channels}, Blocksize: {blocksize})")
            audio_stream = sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype='float32', # mlx-whisper が期待する型
                blocksize=blocksize, # 一度にコールバックに渡されるフレーム数
                callback=audio_callback # 音声データを受け取る関数
            )
            audio_stream.start()
            print("オーディオストリーム開始成功。")

            # 文字起こしスレッド開始
            recording_thread = threading.Thread(target=transcribe_realtime_task,
                                                args=(model_name, output_filepath, self), # selfを渡してGUI操作を可能に
                                                daemon=True)
            recording_thread.start()
            print("リアルタイム文字起こしスレッド開始。")
            self.status_label.configure(text=f"リアルタイム文字起こし中...") # スレッド開始後に更新

        except sd.PortAudioError as pae:
            messagebox.showerror("オーディオエラー", f"マイクの初期化/開始に失敗しました。\nデバイスを確認し、アプリにマイクへのアクセス許可があるか確認してください。\nエラー詳細: {pae}")
            print(f"オーディオストリーム開始エラー: {pae}")
            self.force_stop_realtime() # 開始に失敗したら状態をリセット
        except Exception as e:
            messagebox.showerror("開始エラー", f"リアルタイム文字起こしの開始中に予期せぬエラーが発生しました:\n{e}")
            print(f"リアルタイム文字起こし開始エラー: {e}")
            self.force_stop_realtime()


    def stop_realtime_transcription(self):
        global is_recording, audio_stream, recording_thread, output_file_realtime

        if not is_recording:
            print("停止処理: すでに停止しています。")
            return

        print("リアルタイム文字起こし停止処理を開始...")
        self.status_label.configure(text="停止処理中...")
        self.record_button.configure(state="disabled") # 停止処理中はボタン無効
        self.update()

        # 1. 録音フラグをFalseにして、文字起こしループに停止を通知
        is_recording = False

        # 2. オーディオストリームを停止
        if audio_stream:
            try:
                # ストリームを停止してもコールバックが呼ばれることがあるため、先に止める
                audio_stream.stop()
                audio_stream.close()
                print("オーディオストリーム停止完了。")
            except Exception as e:
                print(f"オーディオストリーム停止中にエラー: {e}")
            finally:
                audio_stream = None # クリア

        # 3. 文字起こしスレッドの終了を待つ (タイムアウト付き)
        if recording_thread and recording_thread.is_alive():
            print("文字起こしスレッドの終了を待機 (最大5秒)...")
            recording_thread.join(timeout=5.0)
            if recording_thread.is_alive():
                print("警告: 文字起こしスレッドが時間内に正常終了しませんでした。")
                # 必要であれば強制終了などの処理だが、基本はjoin()で待つ
            else:
                print("文字起こしスレッド終了確認。")
        recording_thread = None # クリア

        # ファイルは transcribe_realtime_task の finally で閉じられるはずだが念のため
        if output_file_realtime and not output_file_realtime.closed:
             try:
                output_file_realtime.close()
                print("出力ファイル（リアルタイム）を閉じました（念のため）。")
             except Exception as e:
                print(f"出力ファイル（リアルタイム）クローズエラー（念のため）: {e}")
             finally:
                output_file_realtime = None # クリア


        # 4. UIの状態をアイドル状態に戻す
        self.set_ui_state(realtime_recording=False)
        self.status_label.configure(text="リアルタイム文字起こし停止")
        # 最後のメッセージを表示する場合
        self.transcription_textbox.configure(state="normal")
        self.transcription_textbox.insert(tk.END, "\n--- リアルタイム文字起こし終了 ---")
        self.transcription_textbox.see(tk.END)
        self.transcription_textbox.configure(state="disabled")
        print("リアルタイム文字起こし停止処理完了。")

    def force_stop_realtime(self):
        """エラー発生時などに強制的にリアルタイム処理を停止しUIをリセット"""
        print("エラー発生のため、リアルタイム処理を強制停止します。")
        global is_recording, audio_stream, recording_thread, output_file_realtime
        is_recording = False # まずフラグを折る

        if audio_stream:
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception as e:
                print(f"強制停止: オーディオストリーム停止エラー: {e}")
            finally:
                audio_stream = None

        # スレッドの終了は待たない（待つとデッドロックの可能性）
        recording_thread = None
        output_file_realtime = None # ファイルは閉じられないかもしれない

        # UIリセット
        self.set_ui_state(realtime_recording=False)
        self.status_label.configure(text="エラー発生により停止")


    def update_transcription_display(self, text):
        """別スレッドからテキストボックスを安全に更新する"""
        try:
            if self.transcription_textbox.winfo_exists(): # ウィジェットが存在するか確認
                self.transcription_textbox.configure(state="normal")
                self.transcription_textbox.insert(tk.END, text)
                self.transcription_textbox.see(tk.END) # 自動スクロール
                self.transcription_textbox.configure(state="disabled")
                # self.update_idletasks() # 即時反映させたい場合（負荷に注意）
        except tk.TclError as e:
            print(f"GUI更新エラー（ウィンドウが閉じられた可能性）: {e}")
        except Exception as e:
            print(f"GUI更新中に予期せぬエラー: {e}")


# --- アプリケーションの実行 ---
if __name__ == "__main__":
    app = TranscriptionApp()

    # アプリ終了時の処理
    def on_closing():
        print("アプリケーション終了処理を開始...")
        if is_recording:
            print("リアルタイム文字起こしが実行中のため、停止します...")
            # 停止処理を呼び出すが、完了を待たずに終了する可能性あり
            app.stop_realtime_transcription()
            # 少し待機時間を設ける（任意）
            # time.sleep(0.5)
        app.destroy()
        print("アプリケーション終了。")

    app.protocol("WM_DELETE_WINDOW", on_closing) # ウィンドウの閉じるボタンが押されたときの処理
    app.mainloop()
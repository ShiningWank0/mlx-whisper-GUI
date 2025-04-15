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
import traceback # エラー詳細表示用
import requests

# --- グローバル変数 ---
selected_model = "mlx-community/whisper-small-mlx" # ★ デフォルトモデルを 'small' に変更
available_models = [
    "mlx-community/whisper-tiny-mlx", 
    "mlx-community/whisper-base-mlx",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-medium-mlx-fp32",
    "mlx-community/whisper-turbo",
    "mlx-community/whisper-turbo",
    "mlx-community/whisper-large-v3-mlx"
    ]
audio_stream = None
recording_thread = None
is_recording = False # リアルタイム録音中フラグ
output_file_realtime = None
samplerate = 16000
channels = 1
# ★ パフォーマンス関連パラメータ調整
blocksize = 4800       # マイクからの読込単位 (フレーム数, 0.3秒分)
min_chunk_duration = 0.7 # 文字起こしを実行する最小音声長 (秒)
max_chunk_duration = 8.0 # 一度に処理する最大音声長 (秒)
audio_queue = queue.Queue()

# --- 位置情報取得関数 ---
def get_location():
    try:
        geo_request_url = 'https://get.geojs.io/v1/ip/geo.json'
        data = requests.get(geo_request_url).json()
        if data:
            print(f"取得した位置情報: {data['latitude']}, {data['longitude']}")
            return data['latitude'], data['longitude']
        else:
            print("IPベースの位置情報取得に失敗。")
            return "エラー", "エラー"
    except Exception as e:
        print(f"位置情報取得中に予期せぬエラー: {e}")
        messagebox.showerror("位置情報エラー", f"位置情報取得中に予期せぬエラーが発生しました。\n{e}")
        return "エラー", "エラー"

# --- 音声コールバック関数 (リアルタイム録音用) ---
def audio_callback(indata, frames, time, status):
    """マイクから音声データを受け取りキューに入れる"""
    if status:
        print(f"オーディオコールバックステータス: {status}", flush=True)
    audio_queue.put(indata.copy())

# --- リアルタイム文字起こし処理関数 (別スレッドで実行) ---
def transcribe_realtime_task(model_name, output_filename, app_instance):
    global is_recording, output_file_realtime, samplerate, audio_queue, min_chunk_duration, max_chunk_duration

    try:
        # --- 初期設定 ---
        latitude, longitude = get_location()
        if latitude != "エラー" and longitude != "エラー":
            location_info = f"録音開始地点: 緯度 {float(latitude):.6f}, 経度 {float(longitude):.6f}\n---\n"
        else:
            location_info = "録音開始地点: 位置情報取得エラー\n---\n"
        print(f"RT: {location_info.strip()}")
        app_instance.after(0, lambda: app_instance.update_transcription_display(location_info))

        print(f"RT: モデル '{model_name}' をロード中...")
        # モデルの事前ロードは行わない
        # app_instance.after(0, lambda: app_instance.status_label.configure(text=f"モデル '{model_name}' をロード中..."))
        # # model = mlx_whisper.load_model(model_name, verbose=False)
        # # load_models.load_modelを使用するように変更
        # model = mlx_whisper.load_models.load_model(model_name, verbose=False)
        # print(f"RT: モデル '{model_name}' ロード完了。")
        app_instance.after(0, lambda: app_instance.status_label.configure(text=f"リアルタイム文字起こし中 ({model_name})..."))

        # --- ファイルオープンとメインループ ---
        with open(output_filename, "w", encoding="utf-8") as f:
            output_file_realtime = f
            output_file_realtime.write(location_info)
            print(f"RT: 出力ファイル '{output_filename}' オープン。")

            accumulated_audio = np.array([], dtype=np.float32)
            last_process_time = time.time()

            while is_recording: # is_recording フラグが True の間ループ
                process_now = False
                try:
                    # --- 音声データ取得 ---
                    temp_audio_list = []
                    while not audio_queue.empty():
                        try:
                            # 複数まとめて取得
                            num_items = audio_queue.qsize()
                            for _ in range(min(num_items, 10)): temp_audio_list.append(audio_queue.get_nowait())
                        except queue.Empty: break
                    if temp_audio_list:
                        new_audio = np.concatenate([chunk.flatten() for chunk in temp_audio_list])
                        accumulated_audio = np.concatenate((accumulated_audio, new_audio))
                        # print(f"RT: 音声キューから {len(temp_audio_list)} 個取得、現在バッファ {len(accumulated_audio)/samplerate:.2f} 秒")

                    # --- 文字起こし実行判定 ---
                    current_duration = len(accumulated_audio) / samplerate
                    # 十分な長さがあるか、または前回の処理から一定時間経過したら処理を試みる
                    # (短い発話が続いても結果が出るように)
                    time_since_last = time.time() - last_process_time
                    if current_duration >= min_chunk_duration or (current_duration > 0 and time_since_last > max_chunk_duration / 2):
                        process_now = True

                    # --- 文字起こし実行 ---
                    if process_now and len(accumulated_audio) > 0:
                        process_len = min(len(accumulated_audio), int(max_chunk_duration * samplerate))
                        audio_to_process = accumulated_audio[:process_len]
                        accumulated_audio = accumulated_audio[process_len:] # 残りをバッファに

                        print(f"RT: 音声チャンク ({len(audio_to_process)/samplerate:.2f}秒) を処理...")
                        # ★ transcribe関数はモデルオブジェクトを第一引数に取らない可能性？
                        #   test.pyでは audio と path_or_hf_repo を渡している。
                        #   もしエラーになる場合は、model引数を削除し、
                        #   path_or_hf_repo=model_name を渡す形式に変更する必要がある。
                        #   ひとまず現状維持で試す。
                        # result = mlx_whisper.transcribe(model, # ★ ロードしたモデルオブジェクト
                        #                                 audio_to_process, # 音声データ
                        #                                 # path_or_hf_repo=model_name # ★ エラーの場合の代替案
                        #                                 sample_rate=samplerate,
                        #                                 language="ja",
                        #                                 word_timestamps=False,
                        #                                 fp16=True, # ★ 半精度を明示
                        #                                 verbose=False) # 詳細ログ不要ならFalse
                        result = mlx_whisper.transcribe(audio=audio_to_process,
                                                        path_or_hf_repo=model_name,
                                                        # sample_rate=samplerate,
                                                        language="ja",
                                                        word_timestamps=False,
                                                        fp16=True,
                                                        verbose=False)
                        transcript = result["text"].strip()
                        print(f"RT: 結果: '{transcript}'")
                        last_process_time = time.time() # 最終処理時刻を更新

                        if transcript:
                            current_time_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            line_to_write = f"[{current_time_jst}] {transcript}\n"
                            # ★ ファイル書き込みのエラーも考慮 (ただし、頻発する場合は問題)
                            try:
                                output_file_realtime.write(line_to_write)
                                output_file_realtime.flush()
                                print(f"RT: ファイル書き込み: {line_to_write.strip()}")
                            except Exception as io_err:
                                print(f"!!!! RT: ファイル書き込みエラー: {io_err}")
                            # GUI更新
                            app_instance.after(0, lambda text=line_to_write: app_instance.update_transcription_display(text))

                    # --- 停止時の最終処理 ---
                    elif not is_recording and len(accumulated_audio) > 0:
                        # 停止指示後、バッファに残っている最後のデータを処理
                        print(f"RT: 停止指示後、最後の音声データ ({len(accumulated_audio)/samplerate:.2f}秒) を処理...")
                        audio_to_process = accumulated_audio
                        accumulated_audio = np.array([], dtype=np.float32) # バッファクリア

                        # result = mlx_whisper.transcribe(audio=audio_to_process, model=model, sample_rate=samplerate, language="ja", word_timestamps=False, fp16=True, verbose=False)
                        result = mlx_whisper.transcribe(audio=audio_to_process,
                                                        path_or_hf_repo=model_name,
                                                        # sample_rate=samplerate,
                                                        language="ja",
                                                        word_timestamps=False,
                                                        fp16=True,
                                                        verbose=False)
                        transcript = result["text"].strip()
                        print(f"RT: 最終結果: '{transcript}'")
                        if transcript:
                            current_time_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            line_to_write = f"[{current_time_jst}] {transcript}\n"
                            try:
                                output_file_realtime.write(line_to_write)
                                output_file_realtime.flush()
                            except Exception as io_err:
                                print(f"!!!! RT: 最終ファイル書き込みエラー: {io_err}")
                            app_instance.after(0, lambda text=line_to_write: app_instance.update_transcription_display(text))
                        # 最後の処理が終わったらループを抜ける
                        break # ★ whileループを抜ける
                    else:
                        # データが足りず、まだ録音中なら少し待つ
                        time.sleep(0.05) # ★ 待機時間短縮

                except queue.Empty:
                    # 録音中なら待機
                    if is_recording:
                        time.sleep(0.05)
                    elif len(accumulated_audio) > 0:
                        # 録音停止後、バッファにデータがあれば次のループで処理へ
                        continue
                    else:
                        # 録音停止後、バッファも空ならループ終了
                        break # ★ whileループを抜ける

                # ----- ★ ループ内のエラーハンドリング強化 -----
                except Exception as e_inner:
                    print(f"!!!! RT: ループ内で予期せぬエラー発生: {e_inner}")
                    traceback.print_exc() # 詳細なエラー内容を表示
                    # メモリ関連やRuntimeErrorは回復不能としてループを抜ける
                    if isinstance(e_inner, (MemoryError, RuntimeError)):
                        print("!!!! RT: 回復不能なエラーのためリアルタイム処理を中断します。")
                        app_instance.after(0, lambda: messagebox.showerror("重大なエラー", f"リアルタイム処理中に回復不能なエラーが発生しました:\n{e_inner}\n処理を停止します。"))
                        # force_stop_realtime は finally で呼ばれるのでここでは break のみ
                        break # ★ whileループを抜ける
                    else:
                        # その他のエラーはログに残して継続を試みる
                        print("!!!! RT: エラーが発生しましたが、処理の継続を試みます...")
                        # GUIにエラー表示 (任意)
                        # app_instance.after(0, lambda msg=str(e_inner): app_instance.update_transcription_display(f"[ループ内エラー] {msg}\n"))
                        time.sleep(1) # 少し待ってから次のループへ

            # --- while is_recording ループ終了後 ---
            print("RT: is_recording が False になるか、breakによりループを終了しました。")

    # ----- ★ 関数全体のエラーハンドリング (致命的なエラー) -----
    except sd.PortAudioError as pae:
        print(f"!!!! RT: オーディオデバイスエラー (致命的): {pae}")
        app_instance.after(0, lambda: messagebox.showerror("オーディオエラー", f"マイクの初期化またはストリーム中にエラーが発生しました。\n{pae}\n処理を停止します。"))
        app_instance.after(0, app_instance.force_stop_realtime) # 強制停止
    except RuntimeError as rte:
        print(f"!!!! RT: ランタイムエラー (モデルロード失敗など、致命的): {rte}")
        traceback.print_exc()
        app_instance.after(0, lambda: messagebox.showerror("実行時エラー", f"処理中に実行時エラーが発生しました。\nモデルデータや依存関係を確認してください。\n{rte}\n処理を停止します。"))
        app_instance.after(0, app_instance.force_stop_realtime) # 強制停止
    except AttributeError as ae:
        print(f"!!!! RT: 属性エラー (致命的): {ae}")
        traceback.print_exc()
        msg = f"プログラムの属性アクセスエラー:\n{ae}\nライブラリのバージョンやコードを確認してください。"
        # 'load_model'に関するエラーなら専用メッセージ
        if 'load_model' in str(ae):
            msg += "\n\n'load_model'は'mlx_whisper.load_models'内にある可能性があります。"
        app_instance.after(0, lambda: messagebox.showerror("属性エラー", msg + "\n処理を停止します。"))
        app_instance.after(0, app_instance.force_stop_realtime)
    except Exception as e_outer:
        print(f"!!!! RT: タスク全体で予期せぬエラー (致命的): {e_outer}")
        traceback.print_exc()
        app_instance.after(0, lambda: messagebox.showerror("予期せぬエラー", f"リアルタイム文字起こし中に予期せぬエラーが発生しました:\n{e_outer}\n処理を停止します。"))
        app_instance.after(0, app_instance.force_stop_realtime) # 強制停止
    finally:
        # --- 終了処理 ---
        if output_file_realtime and not output_file_realtime.closed:
            try:
                output_file_realtime.close()
                print("RT: 出力ファイルを閉じました。")
            except Exception as e_close:
                print(f"!!!! RT: 出力ファイルクローズエラー: {e_close}")
        output_file_realtime = None
        print("RT: スレッド終了処理完了。")
        # is_recording は stop または force_stop で False になっているはず
        # UIの状態も stop/force_stop 内でリセットされる


# --- GUI アプリケーションクラス ---
class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("文字起こしアプリ (mlx-whisper)")
        self.geometry("700x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # ★ ファイル文字起こし中フラグ
        self.is_file_transcribing = False

        # --- GUIウィジェット作成 ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(pady=10, padx=10, fill="x")
        self.model_label = ctk.CTkLabel(self.control_frame, text="モデル:")
        self.model_label.pack(side="left", padx=(0, 5))
        self.model_var = ctk.StringVar(value=selected_model)
        self.model_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.model_var, values=available_models, command=self.on_model_select)
        self.model_dropdown.pack(side="left", padx=5)
        self.status_label = ctk.CTkLabel(self.control_frame, text="準備完了", width=200)
        self.status_label.pack(side="right", padx=(5, 0))
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
        self.realtime_frame = ctk.CTkFrame(self.main_frame)
        self.realtime_frame.pack(pady=10, padx=10, fill="x")
        self.realtime_title_label = ctk.CTkLabel(self.realtime_frame, text="リアルタイム文字起こし (マイク)", font=ctk.CTkFont(weight="bold"))
        self.realtime_title_label.pack(anchor="w", pady=(0, 5))
        self.record_button = ctk.CTkButton(self.realtime_frame, text="録音開始", command=self.toggle_realtime_transcription, width=120)
        self.record_button.pack(anchor="w")
        self.display_label = ctk.CTkLabel(self.main_frame, text="文字起こし結果:", font=ctk.CTkFont(weight="bold"))
        self.display_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.transcription_textbox = ctk.CTkTextbox(self.main_frame, state="disabled", wrap="word")
        self.transcription_textbox.pack(pady=(5, 10), padx=10, fill="both", expand=True)

    def on_model_select(self, choice):
        # (変更なし)
        global selected_model
        selected_model = choice
        print(f"モデル '{selected_model}' が選択されました。")
        self.status_label.configure(text=f"モデル '{selected_model}' を選択")

    def select_file(self):
        # (変更なし)
        if is_recording:
            messagebox.showwarning("録音中", "リアルタイム文字起こし中はファイルを選択できません。")
            return
        filepath = filedialog.askopenfilename(title="音声ファイルを選択", filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")])
        if filepath:
            self.selected_filepath = filepath
            display_name = os.path.basename(filepath)
            if len(display_name) > 50: display_name = display_name[:20] + "..." + display_name[-27:]
            self.filepath_label.configure(text=display_name)
            self.transcribe_file_button.configure(state="normal") # ファイル選択したらボタン有効化
            self.status_label.configure(text=f"ファイル選択: {os.path.basename(filepath)}")
            self.transcription_textbox.configure(state="normal")
            self.transcription_textbox.delete("1.0", tk.END)
            self.transcription_textbox.insert("1.0", f"ファイル '{os.path.basename(filepath)}' が選択されました。\n「文字起こし開始 (ファイル)」ボタンを押してください。")
            self.transcription_textbox.configure(state="disabled")

    # ★ UI状態管理関数を更新
    def set_ui_state(self, file_transcribing=None, realtime_recording=None):
        """UI要素の有効/無効を一括で設定"""
        # 引数がNoneでない場合のみフラグを更新
        if file_transcribing is not None:
            self.is_file_transcribing = file_transcribing
        # is_recording はグローバル変数を使用

        is_busy = self.is_file_transcribing or is_recording

        # モデル選択
        self.model_dropdown.configure(state="disabled" if is_busy else "normal")
        # ファイル選択ボタン
        self.select_file_button.configure(state="disabled" if is_busy else "normal")
        # ファイル文字起こしボタン
        file_button_state = "disabled"
        if not is_busy and self.selected_filepath:
            file_button_state = "normal"
        self.transcribe_file_button.configure(state=file_button_state)
        # リアルタイム録音ボタン
        record_button_state = "disabled" if self.is_file_transcribing else "normal"
        record_button_text = "録音開始"
        if is_recording:
            record_button_text = "録音停止"
        self.record_button.configure(state=record_button_state, text=record_button_text)

    def start_file_transcription(self):
        if not self.selected_filepath:
            messagebox.showwarning("ファイル未選択", "文字起こしする音声ファイルを選択してください。")
            return
        if self.is_file_transcribing or is_recording: # 念のため二重チェック
            messagebox.showwarning("処理中", "他の処理が実行中です。")
            return

        model_name = self.model_var.get()
        input_filepath = self.selected_filepath
        path_without_ext, _ = os.path.splitext(input_filepath)
        output_filepath = f"{path_without_ext}_transcribe.txt"

        self.status_label.configure(text=f"ファイル文字起こし準備中...")
        self.set_ui_state(file_transcribing=True) # ★ UI更新 (フラグもTrueに)
        self.transcription_textbox.configure(state="normal")
        self.transcription_textbox.delete("1.0", tk.END)
        self.transcription_textbox.insert("1.0", f"ファイル: {os.path.basename(input_filepath)}\nモデル: {model_name}\n文字起こしを開始します...\n\n")
        self.transcription_textbox.configure(state="disabled")
        self.update()

        threading.Thread(target=self.transcribe_file_task, args=(model_name, input_filepath, output_filepath), daemon=True).start()

    def transcribe_file_task(self, model_name, input_filepath, output_filepath):
        """ファイル文字起こしを別スレッドで実行する"""
        try:
            self.after(0, lambda: self.status_label.configure(text=f"モデル '{model_name}' をロード中..."))
            print(f"ファイル: モデル '{model_name}' をロード中...")
            # model = mlx_whisper.load_model(model_name, verbose=False)
            # mlx_whisper.load_models.load_modelを使用する
            model = mlx_whisper.load_models.load_model(model_name, verbose=False)
            print(f"ファイル: モデル '{model_name}' ロード完了。")
            self.after(0, lambda: self.status_label.configure(text=f"ファイル文字起こし中..."))
            print(f"ファイル: '{input_filepath}' の文字起こしを開始...")
            start_time = time.time()
            # ★ 高速化のため fp16=True を指定
            result = mlx_whisper.transcribe(audio=input_filepath, # 音声ファイルのパスを渡す
                                            # model=model, # モデルオブジェクトは渡さない
                                            path_or_hf_repo=model_name, # モデル名を指定
                                            language="ja",
                                            word_timestamps=True,
                                            fp16=True,
                                            verbose=False)
            end_time = time.time()
            print(f"ファイル: 文字起こし完了。処理時間: {end_time - start_time:.2f}秒")
            self.after(0, lambda: self.process_file_transcription_result(result, input_filepath, output_filepath, model_name))
        except AttributeError as ae: 
            print(f"!!!! ファイル: 属性エラー (致命的): {ae}")
            traceback.print_exc()
            msg = f"プログラムの属性アクセスエラー:\n{ae}\nライブラリのバージョンやコードを確認してください。"
            if 'load_model' in str(ae): msg += "\n\n'load_model'は'mlx_whisper.load_models'内にある可能性があります。"
            self.after(0, lambda: messagebox.showerror("属性エラー", msg))
            self.after(0, lambda: self.status_label.configure(text="属性エラー"))
        except Exception as e:
            print(f"!!!! ファイル文字起こし中にエラー発生: {e}")
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("エラー", f"ファイル文字起こし中にエラーが発生しました:\n{e}"))
            self.after(0, lambda: self.status_label.configure(text="エラー発生"))
        finally:
            # ★ 正常終了・異常終了問わずフラグをリセットし、UI状態を更新
            self.after(0, lambda: self.set_ui_state(file_transcribing=False))

    def process_file_transcription_result(self, result, input_filepath, output_filepath, model_name):
        # (変更なし、前回のコードと同じ)
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
                    start = segment.get('start'); end = segment.get('end'); text = segment.get('text', '').strip()
                    if start is not None and end is not None:
                        start_time_str = str(datetime.timedelta(seconds=start)).split('.')[0] + '.' + f"{int(start % 1 * 1000):03d}"
                        end_time_str = str(datetime.timedelta(seconds=end)).split('.')[0] + '.' + f"{int(end % 1 * 1000):03d}"
                        output_content.append(f"[{start_time_str} --> {end_time_str}] {text}")
                    else: output_content.append(f"[タイムスタンプ不明] {text}")
            else: output_content.append("タイムスタンプ情報（セグメント）は利用できませんでした。")
            full_output_text = "\n".join(output_content)
            with open(output_filepath, "w", encoding="utf-8") as f: f.write(full_output_text)
            print(f"ファイル: 結果を '{output_filepath}' に保存しました。")
            self.transcription_textbox.configure(state="normal")
            self.transcription_textbox.insert(tk.END, "\n" + full_output_text)
            self.transcription_textbox.configure(state="disabled")
            self.status_label.configure(text=f"完了: {os.path.basename(output_filepath)}")
            messagebox.showinfo("完了", f"文字起こしが完了し、結果を\n{output_filepath}\nに保存しました。")
        except Exception as e:
            print(f"!!!! ファイル結果処理中にエラー: {e}")
            messagebox.showerror("結果処理エラー", f"文字起こし結果の処理中にエラーが発生しました:\n{e}")
            self.status_label.configure(text="結果処理エラー")

    def toggle_realtime_transcription(self):
        global is_recording
        if is_recording:
            self.stop_realtime_transcription()
        else:
            # ★ ファイル文字起こし実行中は開始しない (フラグでチェック)
            if self.is_file_transcribing:
                 messagebox.showwarning("処理中", "ファイル文字起こし処理が完了するまでお待ちください。")
                 return
            # 念のため is_recording もチェック
            if is_recording:
                print("!!!! 既に録音中のため開始しません。")
                return
            self.start_realtime_transcription()

    def start_realtime_transcription(self):
        global is_recording, audio_stream, recording_thread, audio_queue, samplerate, channels, blocksize

        if is_recording: return # 二重開始防止

        model_name = self.model_var.get()
        now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        output_filename = now_jst.strftime("%Y%m%d_%H%M%S") + "_realtime.txt"
        output_dir = os.getcwd()
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

        # ★ UI更新 (is_recording フラグはここではまだ False)
        self.set_ui_state(realtime_recording=True) # is_recordingはまだFalseだがUIは録音中表示に
        self.status_label.configure(text=f"リアルタイム文字起こし準備中...")
        self.transcription_textbox.configure(state="normal")
        self.transcription_textbox.delete("1.0", tk.END)
        self.transcription_textbox.insert("1.0", f"リアルタイム文字起こしを開始します...\nモデル: {model_name}\n出力ファイル: {output_filename}\n\n")
        self.transcription_textbox.configure(state="disabled")
        self.update()

        while not audio_queue.empty(): # キューをクリア
            try: audio_queue.get_nowait()
            except queue.Empty: break

        try:
            print(f"RT: オーディオストリームを開始 (Rate: {samplerate}, Block: {blocksize})")
            audio_stream = sd.InputStream(
                samplerate=samplerate, channels=channels, dtype='float32',
                blocksize=blocksize, callback=audio_callback
            )
            audio_stream.start()
            print("RT: オーディオストリーム開始成功。")

            is_recording = True # ★ 実際にストリームを開始できたらフラグを True に
            self.set_ui_state() # UI状態を is_recording=True に合わせて再設定

            recording_thread = threading.Thread(target=transcribe_realtime_task,
                                                args=(model_name, output_filepath, self),
                                                daemon=True)
            recording_thread.start()
            print("RT: 文字起こしスレッド開始。")
            self.status_label.configure(text=f"リアルタイム文字起こし中...") # スレッド開始後に表示更新

        except sd.PortAudioError as pae:
             print(f"!!!! RT: オーディオストリーム開始エラー: {pae}")
             messagebox.showerror("オーディオエラー", f"マイクの初期化/開始に失敗しました。\n{pae}")
             is_recording = False # ★ 開始失敗時はフラグを戻す
             self.set_ui_state() # UI状態を戻す
             self.status_label.configure(text="マイク開始エラー")
        except Exception as e:
             print(f"!!!! RT: 開始時に予期せぬエラー: {e}")
             traceback.print_exc()
             messagebox.showerror("開始エラー", f"リアルタイム文字起こしの開始中にエラー:\n{e}")
             is_recording = False # ★ 開始失敗時はフラグを戻す
             self.set_ui_state() # UI状態を戻す
             self.status_label.configure(text="開始エラー")

    def stop_realtime_transcription(self):
        global is_recording, audio_stream, recording_thread, output_file_realtime

        if not is_recording:
            print("RT停止処理: 既に停止しています。")
            return

        print("RT停止処理 開始...")
        self.status_label.configure(text="停止処理中...")
        self.record_button.configure(state="disabled") # 停止処理中はボタン無効
        self.update()

        is_recording = False # ★ ループに停止を通知

        # オーディオストリーム停止
        if audio_stream:
            try:
                audio_stream.stop()
                audio_stream.close()
                print("RT停止処理: オーディオストリーム停止完了。")
            except Exception as e:
                print(f"!!!! RT停止処理: オーディオストリーム停止エラー: {e}")
            finally: audio_stream = None

        # 文字起こしスレッド終了待機
        if recording_thread and recording_thread.is_alive():
            print("RT停止処理: 文字起こしスレッド終了待機 (最大5秒)...")
            recording_thread.join(timeout=5.0)
            if recording_thread.is_alive(): print("!!!! RT停止処理: スレッドが時間内に終了せず。")
            else: print("RT停止処理: スレッド終了確認。")
        recording_thread = None

        # ファイルは transcribe_realtime_task の finally で閉じられる想定

        # UI状態をアイドルに戻す
        self.set_ui_state() # is_recording=False に基づいてUI更新
        self.status_label.configure(text="リアルタイム文字起こし停止")
        self.update_transcription_display("\n--- リアルタイム文字起こし終了 ---\n")
        print("RT停止処理 完了。")

    def force_stop_realtime(self):
        """エラー発生時などに強制的にリアルタイム処理を停止しUIをリセット"""
        global is_recording, audio_stream, recording_thread, output_file_realtime

        if not is_recording: return # 既に止まっていれば何もしない

        print("RT強制停止 実行...")
        is_recording = False # ★ まずフラグを折る

        if audio_stream:
            try: audio_stream.stop(); audio_stream.close()
            except Exception as e: print(f"強制停止: ストリーム停止エラー: {e}")
            finally: audio_stream = None

        recording_thread = None # スレッド終了は待たない
        output_file_realtime = None # ファイルも閉じられない可能性

        # UIリセット
        self.set_ui_state() # is_recording=False に基づいてUI更新
        self.status_label.configure(text="エラー発生により停止")
        print("RT強制停止 完了。")

    def update_transcription_display(self, text):
        # (変更なし)
        try:
            if self.transcription_textbox.winfo_exists():
                self.transcription_textbox.configure(state="normal")
                self.transcription_textbox.insert(tk.END, text)
                self.transcription_textbox.see(tk.END)
                self.transcription_textbox.configure(state="disabled")
        except tk.TclError as e: print(f"GUI更新エラー（ウィンドウ閉鎖?）: {e}")
        except Exception as e: print(f"GUI更新中エラー: {e}")

# --- アプリケーションの実行 ---
if __name__ == "__main__":
    app = TranscriptionApp()

    def on_closing():
        print("アプリケーション終了処理 開始...")
        if is_recording:
            print("リアルタイム文字起こし実行中。停止します...")
            app.stop_realtime_transcription() # 終了時に停止処理を呼ぶ
            # 必要であれば、ここで time.sleep(0.5) など少し待つ
        app.destroy()
        print("アプリケーション終了。")

    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()
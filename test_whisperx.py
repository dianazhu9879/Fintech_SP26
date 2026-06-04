# test_whisperx.py
import whisperx
audio = whisperx.load_audio("./qa_clips/AAPL_2025_10_30_earnings_call_qa.mp3")
model = whisperx.load_model("small", device="cpu", compute_type="int8")
result = model.transcribe(audio, batch_size=8)
print("OK")
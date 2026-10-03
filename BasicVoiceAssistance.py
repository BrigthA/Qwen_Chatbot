import tempfile
import sounddevice as sd
import numpy as np
import whisper
import ollama
from gtts import gTTS
import os
import platform

# Configuration
MODEL_NAME = "qwen2.5:0.5b"
WHISPER_SIZE = "tiny"  # Tiny model is fast and lightweight
SAMPLE_RATE = 16000
DURATION = 5  # Max recording duration per turn in seconds

print("Loading Whisper Speech-to-Text model...")
whisper_model = whisper.load_model(WHISPER_SIZE)
print(f"Ollama will use brain: {MODEL_NAME}")

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print("\n🎤 Listening... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until the recording is finished
    print("🛑 Stopped recording. Processing...")
    return np.squeeze(audio)

def speak_text(text):
    print(f"🤖 Assistant: {text}")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Play audio based on OS
        if platform.system() == "Darwin": # macOS
            os.system(f"afplay {temp_file}")
        elif platform.system() == "Windows":
            os.system(f"start {temp_file}")
        else: # Linux
            os.system(f"mpg123 {temp_file} || ffplay -nodisp -autoexit {temp_file}")
    except Exception as e:
        print(f"Audio playback error: {e}")

def main():
    chat_history = [{"role": "system", "content": "You are a helpful, concise voice assistant. Keep your answers short and conversational."}]
    
    print("\n=== Local Voice Assistant Initialized ===")
    print("Press Ctrl+C in the terminal to exit.")

    while True:
        try:
            # 1. Record user voice
            audio_data = record_audio()

            # 2. Transcribe speech to text using Whisper
            result = whisper_model.transcribe(audio_data, language="en", fp16=False)
            user_text = result["text"].strip()

            if not user_text:
                print("Didn't catch that. Try speaking again.")
                continue

            print(f"👤 You said: {user_text}")

            if user_text.lower() in ["exit", "quit", "stop"]:
                speak_text("Goodbye!")
                break

            # 3. Send text to Ollama (Qwen 0.5B)
            chat_history.append({"role": "user", "content": user_text})
            
            response = ollama.chat(
                model=MODEL_NAME,
                messages=chat_history
            )
            
            assistant_response = response['message']['content']
            chat_history.append({"role": "assistant", "content": assistant_response})

            # 4. Convert response to speech
            speak_text(assistant_response)

        except KeyboardInterrupt:
            print("\nExiting voice assistant. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
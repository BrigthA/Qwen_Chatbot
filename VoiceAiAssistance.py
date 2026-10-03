import streamlit as st
import whisper
import ollama
from gtts import gTTS
import glob
import os
import shutil
import tempfile

# Page configuration
st.set_page_config(
    page_title="Qwen Voice Assistant UI",
    page_icon="🎙️",
    layout="centered"
)

if os.name == "nt" and shutil.which("ffmpeg") is None:
    winget_ffmpeg_dirs = glob.glob(
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg.Essentials_*", "*", "bin"
        )
    )
    for ffmpeg_dir in winget_ffmpeg_dirs:
        if os.path.isfile(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            break

# Load Whisper model (cached so it only loads once)
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

with st.spinner("Loading local Whisper Speech-to-Text model..."):
    whisper_model = load_whisper()

# App Header
st.title("🎙️ Qwen Voice Assistant")
st.caption("Talk to your local Qwen model using Streamlit audio input & text-to-speech output.")

# Sidebar settings
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Model", ["qwen2.5:0.5b", "qwen2.5:1.5b"], index=0)
    
    if st.button("Clear History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there's an associated audio response saved, display player
        if "audio_path" in message and message["audio_path"] and os.path.exists(message["audio_path"]):
            st.audio(message["audio_path"], format="audio/mp3")

# --- Audio Input ---
st.write("### Speak to Assistant:")
audio_source = st.radio("Audio source", ["Record", "Upload"], horizontal=True)
if audio_source == "Record":
    audio_file = st.audio_input("Record your voice query")
else:
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"],
    )

# Process audio input when user records something
if audio_file is not None:
    # Save the recorded audio to a temporary file for Whisper
    audio_suffix = os.path.splitext(getattr(audio_file, "name", "recording.wav"))[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
        tmp_audio.write(audio_file.read())
        audio_path = tmp_audio.name

    with st.spinner("Transcribing your voice..."):
        try:
            result = whisper_model.transcribe(audio_path, language="en", fp16=False)
            user_text = result["text"].strip()
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            user_text = ""
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    if user_text:
        # 1. Append user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2. Query Ollama (Qwen 0.5B)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                stream = ollama.chat(
                    model=model_name,
                    messages=st.session_state.messages,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error connecting to Ollama: {e}"
                message_placeholder.markdown(full_response)

            # 3. Convert assistant response to Speech (TTS)
            response_audio_path = None
            try:
                tts = gTTS(text=full_response, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tts.save(tf.name)
                    response_audio_path = tf.name
                
                # Play audio automatically in the browser chat
                st.audio(response_audio_path, format="audio/mp3", autoplay=True)
            except Exception as tts_err:
                st.warning(f"Could not generate speech audio: {tts_err}")

            # Append assistant reply with audio reference to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio_path": response_audio_path
            })
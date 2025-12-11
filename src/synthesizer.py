import edge_tts
import logging
import asyncio
import os
import tempfile
import sys
import subprocess
from gtts import gTTS
import pyttsx3
from kokoro_onnx import Kokoro
import soundfile as sf

logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, voice="am_echo", rate="+0%", engine="kokoro"):
        self.voice = voice
        self.rate = rate
        self.engine = engine # 'kokoro', 'piper', 'gtts', 'edge', or 'pyttsx3'
        
        # Kokoro settings
        self.kokoro_model = os.path.join(os.getcwd(), "models", "kokoro-v1.0.onnx")
        self.kokoro_voices = os.path.join(os.getcwd(), "models", "voices-v1.0.bin")
        self.kokoro = None
        
        # Auto-download Kokoro models if missing
        if self.engine == "kokoro":
            self._ensure_kokoro_models()

        # Piper settings
        self.piper_model = os.path.join(os.getcwd(), "models", "en_US-lessac-medium.onnx")
        self.piper_binary = os.path.join(os.getcwd(), ".venv", "bin", "piper")

    def _ensure_kokoro_models(self):
        """Downloads Kokoro model files if they don't exist."""
        os.makedirs(os.path.dirname(self.kokoro_model), exist_ok=True)
        
        models = {
            self.kokoro_model: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            self.kokoro_voices: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        }
        
        import requests
        from tqdm import tqdm

        for path, url in models.items():
            if not os.path.exists(path):
                logger.info(f"Downloading {os.path.basename(path)} from {url}...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(path, 'wb') as f, tqdm(
                        desc=os.path.basename(path),
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            bar.update(size)
                    logger.info(f"Downloaded {path}")
                except Exception as e:
                    logger.error(f"Failed to download {path}: {e}")
                    # Clean up partial download
                    if os.path.exists(path):
                        os.remove(path)
                    raise

    async def speak(self, text):
        logger.info(f"Synthesizing: {text}")
        
        # Determine order based on engine preference
        engines = []
        if self.engine == "kokoro":
            engines = [self._speak_kokoro, self._speak_piper, self._speak_gtts, self._speak_pyttsx3]
        elif self.engine == "piper":
            engines = [self._speak_piper, self._speak_kokoro, self._speak_gtts, self._speak_pyttsx3]
        elif self.engine == "edge":
            engines = [self._speak_edge, self._speak_kokoro, self._speak_gtts, self._speak_pyttsx3]
        elif self.engine == "gtts":
            engines = [self._speak_gtts, self._speak_kokoro, self._speak_edge, self._speak_pyttsx3]
        elif self.engine == "pyttsx3":
            engines = [self._speak_pyttsx3, self._speak_kokoro, self._speak_gtts, self._speak_edge]
        else:
             engines = [self._speak_kokoro, self._speak_piper, self._speak_gtts, self._speak_pyttsx3]

        for speak_func in engines:
            try:
                await speak_func(text)
                return
            except Exception as e:
                logger.warning(f"{speak_func.__name__} failed ({e}). Trying next...")
        
        logger.error("All TTS engines failed. No speech output.")

    async def _speak_kokoro(self, text):
        if not os.path.exists(self.kokoro_model) or not os.path.exists(self.kokoro_voices):
            raise FileNotFoundError("Kokoro model files not found.")

        def _generate():
            if self.kokoro is None:
                self.kokoro = Kokoro(self.kokoro_model, self.kokoro_voices)
            
            samples, sample_rate = self.kokoro.create(text, voice=self.voice, speed=1.0, lang="en-us")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_filename = fp.name
                sf.write(temp_filename, samples, sample_rate)
                return temp_filename

        temp_filename = await asyncio.to_thread(_generate)
        await self._play_audio(temp_filename)
        os.remove(temp_filename)

    async def _speak_piper(self, text):
        if not os.path.exists(self.piper_model):
            raise FileNotFoundError(f"Piper model not found at {self.piper_model}")
        
        if not os.path.exists(self.piper_binary):
            raise FileNotFoundError(f"Piper binary not found at {self.piper_binary}")

        def _generate():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_filename = fp.name
            
            cmd = [
                self.piper_binary,
                "--model", self.piper_model,
                "--output_file", temp_filename
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise Exception(f"Piper failed: {stderr}")
                
            return temp_filename

        temp_filename = await asyncio.to_thread(_generate)
        await self._play_audio(temp_filename)
        os.remove(temp_filename)

    async def _speak_edge(self, text):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            temp_filename = fp.name
        
        await communicate.save(temp_filename)
        await self._play_audio(temp_filename)
        os.remove(temp_filename)

    async def _speak_gtts(self, text):
        # gTTS is synchronous, run in thread
        def _generate():
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                tts.save(fp.name)
                return fp.name
        
        temp_filename = await asyncio.to_thread(_generate)
        await self._play_audio(temp_filename)
        os.remove(temp_filename)

    async def _speak_pyttsx3(self, text):
        def _generate():
            engine = pyttsx3.init()
            # Try to set a good voice
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[0].id) 
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_filename = fp.name
            
            # pyttsx3 save_to_file is also synchronous
            engine.save_to_file(text, temp_filename)
            engine.runAndWait()
            return temp_filename

        temp_filename = await asyncio.to_thread(_generate)
        await self._play_audio(temp_filename)
        os.remove(temp_filename)

    async def _play_audio(self, filename):
        player_cmds = []

        if sys.platform == "darwin":
            player_cmds.append(["afplay", filename])
        
        # Add Linux/standard players
        player_cmds.append(["paplay", filename]) # PulseAudio player (worked in preview)
        player_cmds.append(["gst-launch-1.0", "playbin", f"uri=file://{filename}"])
        player_cmds.append(["mpg123", filename])
        player_cmds.append(["ffplay", "-nodisp", "-autoexit", filename])
        player_cmds.append(["aplay", filename]) # For wav files from pyttsx3/piper

        played = False
        for cmd in player_cmds:
            try:
                # Filter out None or empty commands if logic added dynamically
                if not cmd: continue
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.wait()
                if process.returncode == 0:
                    played = True
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Audio player {cmd[0]} failed: {e}")
        
        if not played:
            raise Exception(f"No suitable audio player found. Tried: {[c[0] for c in player_cmds]}")

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        s = Synthesizer(engine="kokoro", voice="am_echo")
        print("Testing Kokoro...")
        await s.speak("Testing Kokoro playback with am_echo.")
        
        print("Testing gTTS...")
        s.engine = "gtts"
        await s.speak("Testing gTTS playback.")

    asyncio.run(main())

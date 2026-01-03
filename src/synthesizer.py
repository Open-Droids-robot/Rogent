import edge_tts
import logging
import asyncio
import os
import tempfile
import sys
import subprocess
import re
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

        # Streaming queues
        self.synthesis_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue()
        self.text_buffer = ""
        self.is_speaking = False
        self._workers_running = False

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

    def _get_kokoro(self):
        if self.kokoro is None:
            self.kokoro = Kokoro(self.kokoro_model, self.kokoro_voices)
        return self.kokoro

    async def start_workers(self):
        """Starts the background workers for synthesis and playback."""
        if not self._workers_running:
            self._workers_running = True
            
            # Pre-load Kokoro so the first sentence is instant
            if self.engine == "kokoro":
                logger.info("Pre-loading Kokoro model...")
                await asyncio.to_thread(self._get_kokoro)
            
            asyncio.create_task(self._synthesis_worker())
            asyncio.create_task(self._playback_worker())
            logger.info("Synthesizer workers started.")

    async def feed_text(self, text_chunk):
        """Feeds text chunks to the synthesizer. Buffers until a sentence is complete."""
        self.text_buffer += text_chunk
        
        # Split by sentence endings (. ? ! or newline) or semi-colons/commas if buffer is long
        # We use a lookbehind to keep the delimiter
        parts = []
        if len(self.text_buffer) > 40:
             # Split on . ? ! , ; : followed by a space
             parts = re.split(r'(?<=[.?!,;:—])\s+', self.text_buffer)
        else:
             parts = re.split(r'(?<=[.?!])\s+', self.text_buffer)
        
        # The last part might be incomplete, so we keep it in the buffer
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    await self.synthesis_queue.put(sentence.strip())
            self.text_buffer = parts[-1]
        elif self.text_buffer.endswith('\n'): # Handle explicit newlines as breaks
             if self.text_buffer.strip():
                 await self.synthesis_queue.put(self.text_buffer.strip())
             self.text_buffer = ""

    async def flush(self):
        """Flushes any remaining text in the buffer."""
        if self.text_buffer.strip():
            await self.synthesis_queue.put(self.text_buffer.strip())
        self.text_buffer = ""

    async def _synthesis_worker(self):
        logger.info("Synthesis worker started.")
        while True:
            text = await self.synthesis_queue.get()
            try:
                logger.info(f"Synthesizing sentence: {text}")
                # Use the existing synthesis logic but return the file path
                # We need to adapt the _speak_* methods to return the path instead of playing
                # For now, we'll wrap the existing logic or refactor slightly
                
                temp_filename = await self._generate_audio(text)
                if temp_filename:
                    await self.playback_queue.put(temp_filename)
            except Exception as e:
                logger.error(f"Synthesis failed for '{text}': {e}")
            finally:
                self.synthesis_queue.task_done()

    async def _playback_worker(self):
        logger.info("Playback worker started.")
        while True:
            filename = await self.playback_queue.get()
            try:
                self.is_speaking = True
                await self._play_audio(filename)
            except Exception as e:
                logger.error(f"Playback failed for {filename}: {e}")
            finally:
                self.is_speaking = False
                if os.path.exists(filename):
                    os.remove(filename)
                self.playback_queue.task_done()

    async def _generate_audio(self, text):
        """Generates audio file for the given text using the selected engine."""
        # Determine order based on engine preference
        engines = []
        if self.engine == "kokoro":
            engines = [self._generate_kokoro, self._generate_piper, self._generate_gtts, self._generate_pyttsx3]
        elif self.engine == "piper":
            engines = [self._generate_piper, self._generate_kokoro, self._generate_gtts, self._generate_pyttsx3]
        elif self.engine == "edge":
            engines = [self._generate_edge, self._generate_kokoro, self._generate_gtts, self._generate_pyttsx3]
        elif self.engine == "gtts":
            engines = [self._generate_gtts, self._generate_kokoro, self._generate_edge, self._generate_pyttsx3]
        elif self.engine == "pyttsx3":
            engines = [self._generate_pyttsx3, self._generate_kokoro, self._generate_gtts, self._generate_edge]
        else:
             engines = [self._generate_kokoro, self._generate_piper, self._generate_gtts, self._generate_pyttsx3]

        for gen_func in engines:
            try:
                return await gen_func(text)
            except Exception as e:
                logger.warning(f"{gen_func.__name__} failed ({e}). Trying next...")
        
        logger.error("All TTS engines failed. No audio generated.")
        return None

    # Refactored generation methods to return filename only
    async def _generate_kokoro(self, text):
        if not os.path.exists(self.kokoro_model) or not os.path.exists(self.kokoro_voices):
            raise FileNotFoundError("Kokoro model files not found.")

        def _generate():
            self._get_kokoro() # Ensure loaded
            
            samples, sample_rate = self.kokoro.create(text, voice=self.voice, speed=1.0, lang="en-us")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_filename = fp.name
                sf.write(temp_filename, samples, sample_rate)
                return temp_filename

        return await asyncio.to_thread(_generate)

    async def _generate_piper(self, text):
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

        return await asyncio.to_thread(_generate)

    async def _generate_edge(self, text):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            temp_filename = fp.name
        
        await communicate.save(temp_filename)
        return temp_filename

    async def _generate_gtts(self, text):
        def _generate():
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                tts.save(fp.name)
                return fp.name
        return await asyncio.to_thread(_generate)

    async def _generate_pyttsx3(self, text):
        def _generate():
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_filename = fp.name
            engine.save_to_file(text, temp_filename)
            engine.runAndWait()
            return temp_filename
        return await asyncio.to_thread(_generate)

    # Legacy speak method for compatibility (blocking-ish)
    async def speak(self, text):
        logger.info(f"Synthesizing (Legacy): {text}")
        await self.feed_text(text)
        await self.flush()
        # Wait for queues to empty? 
        # For legacy behavior, we might want to wait, but since we are moving to streaming,
        # we can just let it run in background.
        # If we need to block until finished:
        # await self.synthesis_queue.join()
        # await self.playback_queue.join()

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
        await s.start_workers()
        
        print("Testing Streaming...")
        await s.feed_text("This is the first sentence. ")
        await asyncio.sleep(1)
        await s.feed_text("And this is the second one, arriving later. ")
        await s.flush()
        
        # Keep alive to let it finish
        await asyncio.sleep(5)

    asyncio.run(main())

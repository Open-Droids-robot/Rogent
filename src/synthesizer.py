import edge_tts
import logging
import asyncio
import os
import tempfile
from gtts import gTTS

logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, voice="en-US-ChristopherNeural", rate="+0%"):
        self.voice = voice
        self.rate = rate

    async def speak(self, text):
        logger.info(f"Synthesizing: {text}")
        
        # Try Edge TTS first
        try:
            await self._speak_edge(text)
            return
        except Exception as e:
            logger.warning(f"Edge TTS failed ({e}). Falling back to gTTS...")
        
        # Fallback to gTTS
        try:
            await self._speak_gtts(text)
        except Exception as e:
            logger.error(f"gTTS failed ({e}). No speech output.")

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

    async def _play_audio(self, filename):
        # Try GStreamer first (Jetson standard)
        player_cmd = ["gst-launch-1.0", "playbin", f"uri=file://{filename}"]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *player_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
        except FileNotFoundError:
            # Try mpg123
            try:
                process = await asyncio.create_subprocess_exec(
                    "mpg123", filename,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.wait()
            except FileNotFoundError:
                logger.error("No audio player found! Please install mpg123 or gstreamer.")

if __name__ == "__main__":
    async def main():
        s = Synthesizer()
        await s.speak("Testing fallback audio.")
    
    asyncio.run(main())

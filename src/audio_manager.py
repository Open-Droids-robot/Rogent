import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, vad_aggressiveness=3):
        """
        Initialize Audio Manager.
        
        Args:
            sample_rate: Audio sample rate (Hz). Must be 8000, 16000, 32000 or 48000 for WebRTC VAD.
            frame_duration_ms: Duration of a frame in ms. Must be 10, 20, or 30.
            vad_aggressiveness: VAD aggressiveness (0-3). 3 is most aggressive (filters out more non-speech).
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.is_speaking = False  # Gating flag: if True, we are robot-speaking, so ignore input
        
        self.stream = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
            
        if self.is_speaking:
            # GATING: If the robot is speaking, we drop the microphone input to prevent self-hearing.
            return

        # WebRTC VAD requires 16-bit PCM mono audio
        # indata is float32 by default from sounddevice, need to convert
        audio_data = (indata.flatten() * 32768).astype(np.int16)
        
        # We might get more or fewer frames than self.frame_size depending on the blocksize
        # But usually we set blocksize to match frame_size
        self.audio_queue.put(audio_data.tobytes())

    def start(self):
        """Start the audio stream."""
        if self.is_running:
            return

        logger.info("Starting Audio Manager...")
        self.is_running = True
        
        # Start input stream
        # blocksize is set to frame_size to ensure we get the exact chunk size VAD expects
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.frame_size,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stop the audio stream."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Audio Manager stopped.")

    def set_speaking_state(self, speaking: bool):
        """
        Set the speaking state. 
        When True, microphone input is ignored (Gating).
        """
        if speaking != self.is_speaking:
            logger.info(f"Speaking state changed to: {speaking}")
            self.is_speaking = speaking

    def get_audio_generator(self):
        """
        Generator that yields speech frames.
        This is where we can implement more complex logic like:
        - buffering
        - triggering only on speech
        """
        while self.is_running:
            try:
                # Get raw bytes from queue
                frame_bytes = self.audio_queue.get(timeout=0.5)
                
                # Check VAD
                # We assume the frame size is correct because of blocksize in InputStream
                if len(frame_bytes) == self.frame_size * 2: # 2 bytes per sample (int16)
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    yield (frame_bytes, is_speech)
                else:
                    logger.warning(f"Incorrect frame size: {len(frame_bytes)}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {e}")
                break

if __name__ == "__main__":
    # Simple test
    manager = AudioManager()
    manager.start()
    try:
        print("Listening... Press Ctrl+C to stop.")
        for frame, is_speech in manager.get_audio_generator():
            if is_speech:
                print(".", end="", flush=True)
            else:
                print("_", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()

from faster_whisper import WhisperModel
import logging
import os

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, model_size="tiny.en", device="cuda", compute_type="float16"):
        """
        Initialize the Transcriber with faster-whisper.
        
        Args:
            model_size: Size of the Whisper model (tiny.en, base.en, small.en, etc.)
            device: Device to run on ('cuda' for Jetson, 'cpu' otherwise)
            compute_type: 'float16' for GPU, 'int8' for CPU.
        """
        logger.info(f"Loading Whisper model: {model_size} on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU if CUDA fails (common in dev environments)
            if device == "cuda":
                logger.warning("Falling back to CPU...")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise e

    def transcribe(self, audio_data):
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio data (numpy array or path to file). 
                        If numpy array, should be float32, 16kHz, mono.
        
        Returns:
            str: Transcribed text.
        """
        segments, info = self.model.transcribe(audio_data, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text + " "
            
        return text.strip()

if __name__ == "__main__":
    # Test
    import numpy as np
    t = Transcriber(device="cpu", compute_type="int8") # Force CPU for test
    # Generate dummy audio (silence) just to check if it runs
    dummy_audio = np.zeros(16000*2, dtype=np.float32)
    print(f"Transcribing silence: '{t.transcribe(dummy_audio)}'")

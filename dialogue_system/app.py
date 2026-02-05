import re
import time
import soxr
import queue
import base64
import logging
import threading
import numpy as np
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit

from clients.tts_client import IndexTTS_VLLM, Cosyvoice_Streaming_VLLM
from clients.llm_client import QwenLLM_stream
from clients.vad_client import TurnTaking
from modules.utils.backchannel_utils import check_backchannel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="frontend")
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600)


# Headers required for SharedArrayBuffer
@app.after_request
def add_security_headers(response):
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


class Config:
    """Global configuration constants."""

    SAMPLE_RATE = 16000
    VAD_POOL_SIZE = 10
    PORT = 55556


class ChatSession:
    """Manages the full lifecycle of a single user session."""

    def __init__(self, client_id, vad_instance):
        self.client_id = client_id
        self.vad = vad_instance
        self.lock = threading.Lock()
        self._stop_event = threading.Event()  # Internal event to signal interruption
        self.is_active = True

        # State for pending message commitment (handling interruption)
        self.pending_message = None
        self.pending_audio_duration = 0.0
        self.pending_start_time = None
        self.interruption_time = None

    @property
    def stop_event(self):
        return self._stop_event

    def interrupt(self):
        """Interrupts current inference or audio playback."""
        self._stop_event.set()
        socketio.emit("stop_audio", {"message": "interrupt"}, room=self.client_id)
        socketio.emit("circle_status", {"status": "LISTENING"}, room=self.client_id)

    def pause(self):
        """Pauses audio playback."""
        socketio.emit("pause_audio", {"message": "pause"}, room=self.client_id)
        socketio.emit("circle_status", {"status": "LISTENING"}, room=self.client_id)

    def reset_interrupt(self):
        """Creates a new stop event for the next processing cycle, leaving the old one set."""
        self._stop_event = threading.Event()


class SessionManager:
    """Thread-safe manager for active chat sessions."""

    def __init__(self):
        self.sessions = {}
        self._lock = threading.Lock()

    def create_session(self, client_id, vad_instance):
        with self._lock:
            session = ChatSession(client_id, vad_instance)
            self.sessions[client_id] = session
            return session

    def get_session(self, client_id) -> ChatSession:
        return self.sessions.get(client_id)

    def remove_session(self, client_id):
        with self._lock:
            return self.sessions.pop(client_id, None)


class VADModelPool:
    """Object pool for VAD instances to optimize memory and startup time."""

    def __init__(self, model_cls, size=Config.VAD_POOL_SIZE):
        self.pool = queue.Queue(maxsize=size)
        logger.info(f"Initializing VAD Pool with {size} instances...")
        for _ in range(size):
            # Initialize instances without specific callbacks (bound during acquisition)
            instance = model_cls(
                status_callback=None, transcription_callback=None, circle_callback=None
            )
            self.pool.put(instance)

    def acquire(self):
        """Retrieve a VAD instance from the pool."""
        return self.pool.get(block=True)

    def release(self, instance):
        """Reset and return the instance back to the pool."""
        if hasattr(instance, "reset"):
            instance.reset()
        # Clear callbacks to prevent memory leaks or stale context
        instance.status_callback = None
        instance.transcription_callback = None
        instance.circle_callback = None
        self.pool.put(instance)


# ==== Global Singleton Initialization ====
vad_pool = VADModelPool(TurnTaking)
session_manager = SessionManager()
llm = QwenLLM_stream()
# tts = Cosyvoice_Streaming_VLLM()
tts = IndexTTS_VLLM()
asr = None  # Placeholder for ASR client if transcription isn't handled within VAD
print("System initialized: VAD Pool, LLM client, TTS client ready.")


def emit_to_room(client_id, event, data):
    """Helper function to safely emit SocketIO messages to a specific room."""
    socketio.emit(event, data, room=client_id)


def pipeline_worker(client_id, audio_segment, sample_rate):
    """
    Main processing pipeline: ASR -> LLM -> TTS.
    Runs in a background thread for each detected utterance.
    """
    session = session_manager.get_session(client_id)
    if not session:
        return

    try:
        # 1. ASR Phase (Automatic Speech Recognition)
        if asr is None:
            # Fallback if ASR is handled by internal TurnTaking module
            asr_text = (
                audio_segment if isinstance(audio_segment, str) else "Voice Detected"
            )
        else:
            asr_text = asr.recognize(audio_segment, sample_rate)

        if not asr_text.strip():
            return

        # # 2. Backchannel Detection (vad already handles this, but double-check here)
        # # Checks for filler words or short backchannels that shouldn't trigger a full response
        # if check_backchannel(asr_text):
        #     emit_to_room(client_id, "resume_audio", {"message": "backchannel detected"})
        #     return

        # Process Pending Message from previous turn if any (before adding new user message)
        with session.lock:
            if session.pending_message:
                final_msg = session.pending_message
                # Check if an interruption occurred during the playback of the previous complete message
                if (
                    session.interruption_time
                    and session.pending_start_time
                    and session.pending_audio_duration > 0
                ):
                    elapsed = max(
                        session.interruption_time - session.pending_start_time, 0
                    )
                    ratio = min(elapsed / session.pending_audio_duration, 1.0)
                    cutoff = int(len(final_msg) * ratio)
                    final_msg = final_msg[:cutoff]
                    logger.info(
                        f"[{client_id}] Previous turn, Ratio: {ratio:.2f}, Truncated: {final_msg}"
                    )
                else:
                    logger.info(f"[{client_id}] Previous turn completed fully.")

                llm.add_message(client_id, "assistant", final_msg)

            # Clear pending state
            session.pending_message = None
            session.pending_audio_duration = 0.0
            session.pending_start_time = None
            session.interruption_time = None

        # 3. Preparation for Response
        session.interrupt()  # Signal previous threads to stop
        session.reset_interrupt()  # Create a fresh event for this new thread

        # Capture the specific stop_event for this execution cycle
        current_stop_event = session.stop_event

        if current_stop_event.is_set():
            return

        logger.info(f"[{client_id}] ASR result: {asr_text}")
        emit_to_room(client_id, "user_transcription", {"text": asr_text})

        llm.add_message(client_id, "user", asr_text)
        llm_reply_gen = llm.generate_with_history(
            client_id, stop_event=current_stop_event
        )
        # logger.info(llm.get_session(client_id))

        # 4. LLM & TTS Streaming Processing
        # Iterate over LLM response chunks and synthesize audio on the fly
        message_to_add = ""
        interrupted = False
        first_emit_time = None
        total_audio_duration = 0.0

        for i, chunk in enumerate(
            llm_reply_gen
            if hasattr(llm_reply_gen, "__iter__") and not isinstance(llm_reply_gen, str)
            else [llm_reply_gen]
        ):
            if current_stop_event.is_set():
                interrupted = True
                break

            logger.info(f"[{client_id}] LLM Chunk: {chunk}")

            # Send LLM text chunk immediately
            emit_to_room(client_id, "text_response", {"text": chunk})

            # Accumulate text before TTS loop to ensure current chunk is considered
            message_to_add += chunk

            # Synthesize text chunk to speech (Iterate over int16 pcm chunks)
            for wav_chunk in tts.synthesize(chunk, streaming=True):
                if current_stop_event.is_set():
                    interrupted = True
                    break

                if first_emit_time is None:
                    first_emit_time = time.time()

                # Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
                # Assuming 24k sample rate, 1 channel, 16-bit (2 bytes) = 48000 bytes/sec
                total_audio_duration += len(wav_chunk) / 48000.0

                emit_to_room(client_id, "audio_chunk", wav_chunk)

            if current_stop_event.is_set():
                interrupted = True
                break

        if interrupted:
            # If interrupted mid-stream, calculate truncation immediately using current time
            if first_emit_time and total_audio_duration > 0:
                elapsed_time = max(session.interruption_time - first_emit_time, 0)
                ratio = min(elapsed_time / total_audio_duration, 1.0)
                cutoff_length = int(len(message_to_add) * ratio)
                message_to_add = message_to_add[:cutoff_length]
                logger.info(
                    f"[{client_id}] Interrupted mid-stream. Ratio: {ratio:.2f}. Truncated message: {message_to_add}"
                )
            else:
                message_to_add = ""

            # Commit immediately as this pipeline execution is dead
            llm.add_message(client_id, "assistant", message_to_add)

            # Ensure no pending state is left over
            with session.lock:
                session.pending_message = None
                session.pending_audio_duration = 0.0
                session.pending_start_time = None
                session.interruption_time = None

        else:
            # Pipeline finished successfully, but user might interrupt later while audio is playing.
            # Do NOT commit to LLM yet. Save to session pending state.
            with session.lock:
                session.pending_message = message_to_add
                session.pending_audio_duration = total_audio_duration
                session.pending_start_time = first_emit_time
                session.interruption_time = None

    except Exception as e:
        logger.error(f"Error in pipeline for {client_id}: {e}", exc_info=True)


# ==== SocketIO Event Handlers ====
@socketio.on("connect")
def handle_connect():
    sid = request.sid
    logger.info(f"New connection request: {sid}")

    def assign_task(client_id):
        """Background task to initialize session and VAD resource."""
        try:
            emit_to_room(
                client_id,
                "vad_loading",
                {"state": "loading", "message": "Acquiring model resources..."},
            )
            instance = vad_pool.acquire()

            # Bind session-specific callbacks to the VAD instance
            instance.status_callback = lambda s, m: emit_to_room(
                client_id, "vad_status", {"state": s, "message": m}
            )
            instance.transcription_callback = lambda t: emit_to_room(
                client_id, "user_transcription", {"text": t}
            )
            instance.circle_callback = lambda s: emit_to_room(
                client_id, "circle_status", {"status": s}
            )

            session_manager.create_session(client_id, instance)

            emit_to_room(client_id, "connect_ack", {"client_id": client_id})
            emit_to_room(
                client_id,
                "vad_loading",
                {"state": "ready", "message": "Model loaded, ready to experience"},
            )
            logger.info(f"Session initialized for {client_id}")
        except Exception as e:
            logger.error(f"Failed to assign model: {e}")

    socketio.start_background_task(assign_task, sid)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    session = session_manager.remove_session(sid)
    if session:
        session.is_active = False
        session.stop_event.set()
        vad_pool.release(session.vad)  # Return VAD instance to pool
    logger.info(f"Client disconnected and resources released: {sid}")


@socketio.on("duplex_stop")
def handle_duplex_stop():
    """Handles manual stop signal from the frontend."""
    sid = request.sid
    session = session_manager.get_session(sid)
    if session:
        if session.interruption_time is None:
            session.interruption_time = time.time()
        session.stop_event.set()
        session.reset_interrupt()
        logger.info(f"Session manually stopped by client: {sid}")


@socketio.on("audio_data")
def handle_audio_data(data, sample_rate):
    """Processes incoming raw binary audio chunks."""
    sid = request.sid
    session = session_manager.get_session(sid)
    if not session:
        return

    # Convert binary buffer to float32 normalized array
    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    if sample_rate != Config.SAMPLE_RATE:
        audio_chunk = soxr.resample(
            audio_chunk, sample_rate, Config.SAMPLE_RATE, quality="VHQ"
        )

    with session.lock:
        # Feed audio into VAD logic
        segment = session.vad.process(audio_chunk)

    # 3. Decision logic based on VAD result
    if segment is not None:
        if isinstance(segment, list) and segment[0] is None:
            # Case: User started speaking while the bot was responding (Barge-in)
            if session.interruption_time is None:
                session.interruption_time = time.time()
            session.interrupt()
        else:
            # Case: A complete utterance is detected
            threading.Thread(
                target=pipeline_worker,
                args=(sid, segment, Config.SAMPLE_RATE),
                daemon=True,
            ).start()


# ==== Static Resource Routing ====
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def serve_file(path):
    return send_from_directory("frontend", path)


if __name__ == "__main__":
    logger.info(f"Server starting on http://localhost:{Config.PORT}")
    socketio.run(app, host="0.0.0.0", port=Config.PORT, debug=False)

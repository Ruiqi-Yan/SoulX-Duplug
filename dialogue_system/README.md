# Full-Duplex Speech Dialogue System

*This directory contains the implementation of a full-duplex speech dialogue system based on SoulX-Duplug.*

## Preparations
Make sure you are in the `SoulX-Duplug/dialogue_system` directory and all the model weights are downloaded to the `SoulX-Duplug/pretrained_models` folder.


### Environment Setup
```bash
conda create -n dialogue-system -y python=3.10.16
conda activate dialogue-system
conda install -y -c conda-forge pynini==2.1.5
sudo apt-get update
sudo apt-get install sox libsox-dev -y
pip install -r requirements.txt
```


### SoulX-Duplug
Install SoulX-Duplug according to the instructions in the [main README](https://github.com/Soul-AILab/SoulX-Duplug/blob/main/README.md).


### LLM
We utilize Qwen2.5-7B-Instruct as the LLM. Please download the model weights to `pretrained_models/Qwen2.5-7B-Instruct`.

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir ../pretrained_models/Qwen2.5-7B-Instruct

# If you are in mainland China
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ../pretrained_models/Qwen2.5-7B-Instruct
```


### TTS
Currently, we utilize two excellent open-source projects as our TTS models: [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm) and [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice).


- For IndexTTS-vLLM, please refer to [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm) for environment setup and model download.

    ```bash
    cd modules/index_tts_vllm
    conda create -n index-tts-vllm python=3.12
    conda activate index-tts-vllm
    pip install -r requirements.txt
    modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ../pretrained_models/Index-TTS-1.5-vLLM
    ```

- For Async CosyVoice, you can refer to [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice) for model download. Our dialogue-system environment already includes the necessary dependencies.

    ```bash
    huggingface-cli download --resume-download swulling/CosyVoice2-0.5B-vllm --local-dir ../pretrained_models/CosyVoice2-0.5B
    cp -r modules/CosyVoice/async_cosyvoice/CosyVoice2-0.5B/* ../pretrained_models/CosyVoice2-0.5B/
    cd modules/CosyVoice/async_cosyvoice/runtime/async_grpc
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
    ```


## Run the Service

### Launch TTS Server
```bash
conda activate ...  # index-tts-vllm for IndexTTS-vLLM, dialogue-system for Async CosyVoice
bash scripts/tts_server.sh
```

### Launch LLM Server
```bash
conda activate soulx-duplug
bash scripts/llm_server.sh
```

### Launch VAD Server
```bash
conda activate soulx-duplug
bash scripts/vad_server.sh
```

### Launch Dialogue System
```bash
conda activate dialogue-system
bash deploy.sh
```


## Todo List



## Acknowledgment

Great thank is given to QwenLM, CosyVoice, Async CosyVoice, IndexTTS, IndexTTS-vLLM, ChatTTS, and X-Talk for their open-source contribution.

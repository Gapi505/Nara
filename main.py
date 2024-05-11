from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import pipeline
from pyannote.audio import Model
from llama_cpp import Llama
import numpy as np
import asyncio
import dotenv
import torch
import time
import yaml
import sys
import gc
import os

dotenv.load_dotenv()


prompt_template = None
system_prompt_unconstructed = None
with open("templates/prompt_template.yaml", "r") as f:
    prompt_template = yaml.safe_load(f)
with open("templates/system_prompt.yaml", "r") as f:
    system_prompt_unconstructed = yaml.safe_load(f)

conversation = ""


llm = None
transcriber = None
vad = None


def free_memory():
    global llm, transcriber, vad
    
    llm = None
    transcriber = None
    vad = None
    gc.collect()
    torch.cuda.empty_cache()

# Loads the model
def load_llm():
    global llm
    
    if llm is None:
        llm = Llama.from_pretrained(
            "PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed",
            filename="*Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True,
        )

def load_transcriber():
    global transcriber
    transcriber = pipeline(
        "automatic-speech-recognition", model="openai/whisper-tiny.en", device="cuda"
    )

def load_vad():
    global vad
    auth_token = os.getenv("SEGMENTATION_SEGMENTATION_TOKEN")
    vad_model = Model.from_pretrained(
                "pyannote/segmentation-3.0", 
                use_auth_token=auth_token)
    vad = VoiceActivityDetection(segmentation=vad_model)
    HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.25,
    }
    vad.instantiate(HYPER_PARAMETERS)

def construct_prompt(conversation):
    with open("templates/system_prompt.yaml", "r") as f:
        system_prompt_unconstructed = yaml.safe_load(f)
    config = system_prompt_unconstructed["config"].split(" ")

    system_prompt = ""
    for conf in config:
        system_prompt += system_prompt_unconstructed[conf] + "\n"
    
    conversation = conversation.rstrip("\n")
    system_prompt = system_prompt.rstrip("\n")
    
    prompt = prompt_template["prompt"].format(system_prompt=system_prompt,conversation=conversation)

    return prompt

def construct_message(user, message):
    fmessage = prompt_template["message"].format(user=user, message=message)
    return fmessage


async def transcribe(chunk_length_s=100.0, stream_chunk_s=0.25, stop_pause_time=0.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    speaking_start = None
    for transcription in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        
        sys.stdout.write("\033[K")
        print(transcription["text"], end="\n")
        
        # End tests
        chunk = next(mic)
        audio_data = chunk['raw']
        waveform = audio_data / np.max(np.abs(audio_data))
        waveform_tensor = torch.from_numpy(waveform).float().reshape(1, -1)
        segments = vad({"waveform": waveform_tensor, "sample_rate": chunk['sampling_rate']}).get_timeline().support()
        print(segments)
        if speaking_start is None:
            speaking_start = time.time()
        if len(segments) > 0:
            elapsed_time = time.time() - speaking_start
            print(elapsed_time)
            last_segment = segments[-1]
            end_time = last_segment.end
            if elapsed_time - end_time > stop_pause_time:
                break



        if not transcription["partial"][0]:
            break

    return transcription["text"]

async def llm_response(prompt):
    kwargs = {
        "max_tokens": 128,
        "temperature": 0.8,
    }
    if llm is None:
        load_llm()
    print(prompt)
    response = llm(prompt, **kwargs,)
    print(response)
    return response

load_transcriber()
load_llm()
load_vad()

print(torch.cuda.memory_summary(device="cuda", abbreviated=False) )

async def main_loop():
    global conversation
    transcriptionJob = asyncio.create_task(transcribe())
    print("Start conversation")
    while True:
        transcription = await transcriptionJob
        print(transcription)
        message = construct_message("user", transcription)
        conversation += message + "\n"
        prompt = construct_prompt(conversation)
        print(prompt)
        response = await llm_response(prompt)
        response = response["choices"][0]["text"]
        print(response)
        conversation += construct_message("assistant", response) + "\n"
        transcriptionJob = asyncio.create_task(transcribe())
        #print(uncertan_message)

asyncio.run(main_loop())
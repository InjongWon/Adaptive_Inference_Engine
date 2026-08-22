# deploying vLLM mechanism
import modal


MODEL_NAME = "Qwen/Qwen3-1.7B"
VLLM_PORT = 8000
N_GPU = 1

app = modal.App("adaptive-llm-serving-vllm")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

hf_cache = modal.Volume.from_name(
    "huggingface-cache",
    create_if_missing=True,
)

vllm_cache = modal.Volume.from_name(
    "vllm-cache",
    create_if_missing=True,
)


@app.server(
    image=image,
    gpu="B200:1",
    port=8000,
    startup_timeout=900,
    scaledown_window=300,
    target_concurrency=64,
    unauthenticated=True,
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            "float16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "4096",
        ]

        print("Starting vLLM:", " ".join(cmd), flush=True)

        self.process = subprocess.Popen(cmd)

        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM exited during startup with code {return_code}"
            )

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            
            
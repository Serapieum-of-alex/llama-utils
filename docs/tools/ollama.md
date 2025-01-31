# Running a Local LLM using Ollama

- Ollama is a tool for running large language models (LLMs) locally on your machine. It allows you to download, run,
  and interact with various LLMs without relying on cloud-based APIs.
- Ollama is designed for privacy, performance, and ease of use, making it ideal for offline inference.

Key Features of Ollama
- Local Execution – Runs directly on your computer without requiring an internet connection.
- Supports Multiple Models – Includes Mistral, LLaMA 2, Gemma, and more.
- GPU Acceleration – Uses CUDA (NVIDIA) or Metal (Mac) for faster performance.
- API Interface – Exposes a simple HTTP API for easy integration with applications.
- CLI-Based Interaction – Allows running inference from the command line.

## 1. Installation of Ollama

### **Windows**

1. **Download the Installer**
   Visit [Ollama's website](https://ollama.com) and download the Windows installer.
2. **Run the Installer**

   - Double-click the downloaded `.exe` file.
   - Follow the installation prompts.
   - Restart your terminal after installation.
3. **Verify Installation**
   Open **PowerShell** or **Command Prompt** and run:

   ```sh
   ollama --version
   ```

   This should return the installed version.

### **macOS**

1. **Install via Homebrew**

   ```sh
   brew install ollama
   ```
2. **Verify Installation**

   ```sh
   ollama --version
   ```

### **Linux**

1. **Install via Curl**

   ```sh
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Verify Installation**

   ```sh
   ollama --version
   ```

---

## 2. Downloading and Managing Models

### **List Available Models**

```sh
ollama list
```

### **Download a Model**

```sh
ollama pull mistral
ollama pull gemma
ollama pull llama2
```

### **Remove a Model**

```sh
ollama rm mistral
```

---

## 3. Running Ollama Locally

### **Start the Ollama Server**

Run:

```sh
ollama serve
```

Check if the server is running:

```sh
ps aux | grep ollama
```

for Windows use:

```sh
tasklist | findstr /i "ollama"
```

### **Configure GPU Acceleration**

If your system supports CUDA, Ollama will use the GPU automatically. You can check GPU usage via:

```sh
nvidia-smi
```

To force CPU usage:

```sh
OLLAMA_USE_CPU=1 ollama run mistral
```
for Windows use:

```sh  
set OLLAMA_USE_CPU=1
ollama run mistral  
```

### **Troubleshooting**

- If `ollama` commands are not found, restart your shell or check your `$PATH`:
```sh
echo $PATH
```
- For permission issues, try running:
```sh
sudo ollama serve
```

---

## 4. Interacting with Ollama via CLI

### **Basic Inference**

```sh
ollama run mistral
```

Example:

```sh
ollama run mistral "What is the capital of France?"
```

### **Example Response**

```
Paris is the capital of France.
```

---

## 5. Using Ollama as an API

### **Starting the API**

Run:

```sh
ollama serve
```

### **Making a Request via `curl`**

```sh
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Tell me a joke"
}'
```
for command terminal in Windows use:
```sh
curl -X POST "http://127.0.0.1:11434/api/generate" -H "Content-Type: application/json" -d "{\"model\": \"mistral\", \"prompt\": \"Tell me a short fact about AI\", \"options\": {\"max_tokens\": 50}}"
```
### **Using Python (`requests`)**

```python
import requests

url = "http://localhost:11434/api/generate"
data = {"model": "mistral", "prompt": "Tell me a joke"}

response = requests.post(url, json=data)
print(response.json())
```

---

## 6. Integrating Ollama with `llama-index`

### **Install Dependencies**

```sh
pip install llama-index-llms-ollama
```

### **Configuring `llama-index`**

```python
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

llm = Ollama(model="mistral")
Settings.llm = llm
```

### **Querying Ollama via `llama-index`**

```python
response = llm.complete("What is the capital of the Netherlands?")
print(response)
```

```text
The capital city of the Netherlands is Amsterdam. However, it's important to note that The Hague (Den Haag) serves as the seat of government, hosting several key institutions such as the Dutch Parliament and the Supreme Court. Amsterdam, on the other hand, is known for its vibrant culture and economic significance.
```

---

## 7. Performance Optimization

### **GPU Acceleration**

- Ensure you have CUDA installed:
```sh
nvidia-smi
```
- Run Ollama with GPU support:
```sh
OLLAMA_USE_CUDA=1 ollama serve
```
for Windows use:
```sh
set OLLAMA_USE_CUDA=1
ollama serve
```
---

## 8. Common Issues & Troubleshooting


| Issue               | Solution                                 |
| ------------------- | ---------------------------------------- |
| Command not found   | Restart the terminal, check`$PATH`.      |
| GPU not used        | Check`nvidia-smi`, install CUDA drivers. |
| API not responding  | Ensure`ollama serve` is running.         |

---

This guide provides everything needed to run Ollama locally and integrate it with `llama-index`. Let me know if you need further customization! 🚀


## 9.Errors

You might face the following error when you run the `ollama serve` command
```bash
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
This error is due to the port `11434` is already in use, to solve this error, you can check which process is using this port by running the following command
```bash
netstat -ano | findstr :11434
```
for linux users, you can use the following command
```bash
netstat -ano | grep :11434
```

You will get the following output
```bash
    TCP    127.0.0.1:11434        0.0.0.0:0              LISTENING       20796
```
Then you can kill the process by running the following command
```bash
taskkill /F /PID 20796
```
for linux users, you can use the following command
```bash
kill -9 20796
```

Then
you will gee the following output
```bash
SUCCESS: The process with PID 20796 has been terminated.
```

- Then you can run the `ollama serve` command again, you should see the following output
```bash
2024/11/22 23:20:04 routes.go:1189: INFO server config env="map[CUDA_VISIBLE_DEVICES: GPU_DEVICE_ORDINAL: HIP_VISIBLE_DEVICES: HSA_OVERRIDE_GFX_VERSION: HTTPS_PROXY: HTTP_PROXY: NO_PROXY: OLLAMA_DEBUG:false OLLAMA_FLASH_ATTENTION:false OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://127.0.0.1:11434 OLLAMA_INTEL_GPU:false OLLAMA_KEEP_ALIVE:5m0s OLLAMA_LLM_LIBRARY: OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MODELS:C:\\Users\\eng_m\\.ollama\\models OLLAMA_MULTIUSER_CACHE:false OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:0 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://*] OLLAMA_SCHED_SPREAD:false OLLAMA_TMPDIR: ROCR_VISIBLE_DEVICES:]"
time=2024-11-22T23:20:04.393+01:00 level=INFO source=images.go:755 msg="total blobs: 28"
time=2024-11-22T23:20:04.395+01:00 level=INFO source=images.go:762 msg="total unused blobs removed: 0"
time=2024-11-22T23:20:04.397+01:00 level=INFO source=routes.go:1240 msg="Listening on 127.0.0.1:11434 (version 0.4.1)"
time=2024-11-22T23:20:04.400+01:00 level=INFO source=common.go:49 msg="Dynamic LLM libraries" runners="[cpu cpu_avx cpu_avx2 cuda_v11 cuda_v12 rocm]"
time=2024-11-22T23:20:04.400+01:00 level=INFO source=gpu.go:221 msg="looking for compatible GPUs"
time=2024-11-22T23:20:04.400+01:00 level=INFO source=gpu_windows.go:167 msg=packages count=1
time=2024-11-22T23:20:04.400+01:00 level=INFO source=gpu_windows.go:214 msg="" package=0 cores=8 efficiency=0 threads=16
time=2024-11-22T23:20:04.592+01:00 level=INFO source=types.go:123 msg="inference compute" id=GPU-04f76f9a-be0a-544b-9a6f-8607b8d0a9ab library=cuda variant=v12 compute=8.6 driver=12.6 name="NVIDIA GeForce RTX 3060 Ti" total="8.0 GiB" available="7.0 GiB"
```

you can change the port by running the following command
`ollama serve --port 11435`

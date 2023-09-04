
https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML/discussions/7

You only need one file. q4_0 and q4_1 are different models, and you can use either but don't need both.

As for the different sizes: the larger the file the better the accuracy, but the more resources required and the slower the speed. q4_0 and q5_0 are good compromises.

The formats with the letter K in their name are a new type of quantisation. They're generally better than the old types, but don't yet have as wide support. They don't yet work in text-generation-webui for example, I think.

As you only have 16GB RAM, I would limit yourself to 13B models in q4_0 for now.

FYI, llama.cpp recently added Metal acceleration which should give you much better performance. Again I don't think that's yet supported in text-generation-webui, but I would expect it to come in the next week or so. It won't enable you to run larger models (you're still limited by that 16GB), but it will give you much faster performance.


---
LLM Benchmarks:

`garage-bAInd/Platypus2-70B-instruct` is a merge of garage-bAInd/Platypus2-70B and upstage/Llama-2-70b-instruct-v2.

https://huggingface.co/models?sort=trending&search=platypus2-70B-instruct

[garage-bAInd/Platypus2-70B-instruct](https://huggingface.co/garage-bAInd/Platypus2-70B-instruct)

[TheBloke/Platypus2-70B-Instruct-GGML](https://huggingface.co/TheBloke/Platypus2-70B-Instruct-GGML)

[TheBloke/Platypus2-70B-Instruct-GPTQ](https://huggingface.co/TheBloke/Platypus2-70B-Instruct-GPTQ)




https://gpt4all.io/index.html

https://evalgpt.ai/

https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

| Model                                      | Average ⬆️ | ARC   | HellaSwag | MMLU  | TruthfulQA |
|--------------------------------------------|------------|-------|-----------|-------|------------|
| garage-bAInd/Platypus2-70B-instruct        | 73.13      | 71.84 | 87.94     | 70.48 | 62.26      |
| upstage/Llama-2-70b-instruct-v2             | 72.95      | 71.08 | 87.89     | 70.58 | 62.25      |
| deepnight-research/llama-2-70B-inst         | 72.95      | 71.08 | 87.89     | 70.58 | 62.25      |
| psmathur/model_007                         | 72.72      | 71.08 | 87.65     | 69.04 | 63.12      |
| psmathur/orca_mini_v3_70b                  | 72.64      | 71.25 | 87.85     | 70.18 | 61.27      |



**Others:**


Nous-Hermes-Llama-2 13b released, beats previous model on all benchmarks, and is commercially usable.

Refs:
https://www.reddit.com/r/LocalLLaMA/comments/155wwrj/noushermesllama2_13b_released_beats_previous/?rdt=34143

https://gpt4all.io/index.html

https://huggingface.co/NousResearch
https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b/tree/main


https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML/tree/main
https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGML/tree/main


---
---

### Step to run after setting up:
In order to run Llama with GPU, make sure do:
```
pip uninstall llama-cpp-python

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose
```

1. Open terminal at root directory of the project, run:
```
poetry shell
```
2. Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Run the following command to ingest all the data.

```shell
python ingest.py
```

3. Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.


**Notes:**
- Command line to solve error CUDA:
[Issue]: CUDA initialization: Unexpected error from cudaGetDeviceCount().

```
sudo apt install nvidia-cuda-toolkit
nvcc --version
sudo apt update
nvidia-smi
sudo ubuntu-drivers autoinstall
nvidia-smi
python privateGPT.py


CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python
```

Make sure that `nvidia-smi`command output is working fine.


---
---

### References and Troublshooting of setting up

Is privateGPT based on CPU or GPU? Why in my case it's unbelievably slow?

https://github.com/imartinez/privateGPT/issues/931#issuecomment-1668367407

https://github.com/imartinez/privateGPT/issues/885


GPT4All I think is CPU only. At top of their repo (https://github.com/nomic-ai/gpt4all) they say "Open-source assistant-style large language models that run locally on your CPU" which is great for enabling literally anyone to get in on it, but not for GPU people. I could be wrong tho maybe there is some GPU support
If you do use a GPU, you can use ggml models with llama-cpp-python in the way I offer.



https://github.com/imartinez/privateGPT/issues/885#issuecomment-1646752174

IF it doesn't detect your cuda toolkit, it will just build for CPU.

```cmd
pip uninstall llama-cpp-python

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose

Other option:
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose
---

Sample query:

Enter a query: why was the NATO created?

---
Poetry install of Sentence_tranformers is incomplete
https://github.com/imartinez/privateGPT/issues/915


---
**How to install CUDA & cuDNN on Ubuntu 22.04**

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu


https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202


Just do until check: nvcc --version

---

```
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH


export PATH="$PATH:$HOME/.local/bin"
```

---

You can easily check your Ubuntu distribution, architecture, and version using various terminal commands. Here are a few methods to do so:

1. **Using the `lsb_release` Command:**
   Open a terminal and run the following command to get information about your Ubuntu distribution:
   ```bash
   lsb_release -a
   ```

   This command will display information about your distribution name, release number, codename, and more.

2. **Using the `uname` Command:**
   The `uname` command can provide information about your system's architecture. Run the following command:
   ```bash
   uname -m
   ```

   The output will show your machine's architecture, which will typically be either `x86_64` for 64-bit systems or `i686` for 32-bit systems.

3. **Using the `arch` Command:**
   Another simple way to determine your system's architecture is by using the `arch` command:
   ```bash
   arch
   ```

   The output will again display your system's architecture, such as `x86_64` or `i686`.

4. **Using the `lsb_release` Command for Version:**
   To specifically check your Ubuntu version, you can use the `lsb_release` command with the `-r` option:
   ```bash
   lsb_release -r
   ```

   This will show the release version of your Ubuntu distribution.

5. **Using the `/etc/os-release` File:**
   You can also directly view the contents of the `/etc/os-release` file using a command like `cat` to get detailed information about your distribution:
   ```bash
   cat /etc/os-release
   ```

   Look for the `VERSION_ID` field to find the version number.

Keep in mind that when installing software or performing system-related tasks, it's important to have accurate information about your distribution, architecture, and version. If you're not sure about a specific detail, using a combination of these methods will help you gather the necessary information.

---- 

Installing CUDA on Ubuntu involves a series of steps to ensure that your system is properly set up for GPU-accelerated tasks. Here's a general guide on how to install CUDA on Ubuntu:

1. **Check GPU Compatibility:** Before installing CUDA, ensure that your GPU is compatible with the CUDA toolkit. Visit the official NVIDIA website to find a list of GPUs that are supported by the version of CUDA you plan to install.

2. **Update System Packages:**
   Open a terminal and run the following commands to update your system's package list and upgrade existing packages:
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

3. **Install Dependencies:**
   CUDA requires some development tools and libraries. Install them using the following command:
   ```bash
   sudo apt install build-essential
   ```

4. **Download CUDA Toolkit:**
   Visit the NVIDIA CUDA Toolkit download page: https://developer.nvidia.com/cuda-downloads
   Choose your Linux distribution, architecture, and version, and download the installer (`.deb` file).

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

5. **Install CUDA Toolkit:**
   Navigate to the directory where you downloaded the CUDA installer and use the following command to install it:
   ```bash
   sudo dpkg -i cuda-repo-<version>_amd64.deb
   sudo apt update
   sudo apt install cuda
   ```

6. **Set Environment Variables:**
   After the installation is complete, you need to set up some environment variables to allow your system to find CUDA libraries and tools. Add the following lines to your shell profile configuration (e.g., `~/.bashrc` or `~/.zshrc`):
   ```bash
   export PATH=/usr/local/cuda-<version>/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64:$LD_LIBRARY_PATH
   ```
   Replace `<version>` with the appropriate CUDA version you installed.

7. **Verify Installation:**
   After setting the environment variables, save the file and apply the changes by either restarting your terminal or running `source ~/.bashrc` (or the appropriate command for your shell). To verify that CUDA is installed correctly, you can run:
   ```bash
   nvcc --version
   ```

   You should see the version of CUDA that you installed.

Keep in mind that this is a general guide, and there might be small variations based on the CUDA version and your system's configuration. Always refer to the official NVIDIA documentation and resources for the most accurate instructions and troubleshooting.

Please note that the installation process can sometimes be complex, and if you're not familiar with system-level operations, consider seeking assistance from someone experienced or consulting more detailed guides specific to your setup.

----

Error:

python privateGPT.py
CUDA error 35 at /tmp/pip-install-pjr2jjsa/llama-cpp-python_78ae4492f3024aad8fd2ca7b88dba6a8/vendor/llama.cpp/ggml-cuda.cu:4883: CUDA driver version is insufficient for CUDA runtime version


Solutions:
1- run below and see the version of CUDA runtime (in our case: NVIDIA-SMI 470.199.02   Driver Version: 470.199.02   CUDA Version: 11.4)

```
nvidia-smi
```

2. run and see the the version of CUDA compiler (in our case: Cuda compilation tools, release 12.2, V12.2.128 Build cuda_12.2.r12.2/compiler.33053471_0)

```
nvcc --version
```

3. reinstall CUDA toolkits or make sure CUDA compiler is the same as CUDA runtime (in our case: 11.4)
Refs:

https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one

https://stackoverflow.com/questions/56431461/how-to-remove-cuda-completely-from-ubuntu




3.1- Reinstall nvidia-smi or CUDA runtime

```
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*
sudo apt update && sudo apt upgrade
```

```
# sudo ubuntu-drivers autoinstall
ubuntu-drivers devices
sudo apt install nvidia-driver-535
nvidia-smi
reboot
```

Notes:
- Run nvidia-smi will show: NVIDIA-SMI 535.86.05              Driver Version: 535.86.05    CUDA Version: 12.2 


3.2- Reinstall nvcc or CUDA toolkit or compiler

```
sudo apt-get remove nvidia-cuda-toolkit
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
```

Refs:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

```
wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
sudo sh cuda_12.2.1_535.86.10_linux.run
nvcc --version
```

Notes:
- When run sudo sh cuda_12.2.1_535.86.10_linux.run, select only Toolkit to install toolkit only.


-----

Error:
(privategpt-py3.10) (base) connectedin@connectedin:~/Desktop/privateGPT$ python privateGPT.py
/home/connectedin/.cache/pypoetry/virtualenvs/privategpt-48b7A2Xg-py3.10/lib/python3.10/site-packages/torch/cuda/__init__.py:107: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
CUDA error 999 at /tmp/pip-install-pjr2jjsa/llama-cpp-python_78ae4492f3024aad8fd2ca7b88dba6a8/vendor/llama.cpp/ggml-cuda.cu:4883: unknown error

Solution:
https://discuss.pytorch.org/t/userwarning-cuda-initialization-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set-up-environment-e-g-changing-env-variable-cuda-visible-devices-after-program-start-setting-the-available-devices-to-be-zero/129335/8

Anyone still who still has this issue try:

sudo apt-get install nvidia-modprobe
worked for me!
Source: RuntimeError: CUDA unknown error · Issue #49081 · pytorch/pytorch · GitHub 



### LlamaCpp

-----
-----

Sure, I can explain the parameters in the `LlamaCpp` constructor:

* `model_path`: The path to the pre-trained LLM model.
* `n_gqa`: The number of GQA questions to use for fine-tuning the LLM. This parameter is only needed for the LLMa 70b model.
* `n_ctx`: The maximum context length for the LLM. This is the number of tokens that the LLM can process at once.
* `max_tokens`: The maximum number of tokens to generate. This is usually set to the same value as `n_ctx`.
* `n_batch`: The batch size for training and inference.
* `callbacks`: A list of callbacks to be used during training.
* `n_gpu_layers`: The number of LLM layers to use on the GPU.
* `verbose`: Whether to print verbose output during training.

The default values for most of these parameters are reasonable, but you may need to adjust them depending on the specific LLM model and the task you are using it for.

Here are some additional details about each parameter:

* `n_gqa`: The number of GQA questions to use for fine-tuning the LLM is only needed for the LLMa 70b model. The GQA dataset is a large dataset of question-answer pairs, and fine-tuning the LLM on this dataset can improve its performance on question answering tasks.
* `n_ctx`: The maximum context length for the LLM is the number of tokens that the LLM can process at once. This is important to set correctly, as setting it too low can limit the LLM's ability to understand long sequences of text, while setting it too high can waste memory and computational resources.
* `max_tokens`: The maximum number of tokens to generate is usually set to the same value as `n_ctx`. This is because the LLM will only generate up to `n_ctx` tokens, even if you specify a longer number.
* `n_batch`: The batch size for training and inference is the number of examples that are processed at once. A larger batch size can improve the performance of the LLM, but it can also require more memory and computational resources.
* `callbacks`: A list of callbacks to be used during training can be used to monitor the training progress and take corrective action if necessary. For example, you could use a callback to stop training if the LLM's performance starts to plateau.
* `n_gpu_layers`: The number of LLM layers to use on the GPU can be used to improve the performance of the LLM, but it can also require more powerful hardware.
* `verbose`: Whether to print verbose output during training can be helpful for debugging and troubleshooting.

I hope this explanation is helpful. Please let me know if you have any other questions.

------

The parameters that affect the speed and accuracy of the LLM are:

* **Batch size:** A larger batch size can improve the accuracy of the LLM, but it can also slow down training.
* **Number of GPU layers:** Using more GPU layers can improve the speed of training, but it also requires more powerful hardware.
* **Maximum context length:** A larger maximum context length can improve the accuracy of the LLM, but it can also slow down training and require more memory.

For speed, you should use a smaller batch size and fewer GPU layers. You should also set the maximum context length to a smaller value.

For accuracy, you should use a larger batch size and more GPU layers. You can also set the maximum context length to a larger value.

The following table summarizes the recommended values for the parameters for speed and accuracy:

| Parameter | Speed | Accuracy |
|---|---|---|
| Batch size | Small | Large |
| Number of GPU layers | Fewer | More |
| Maximum context length | Small | Large |

Here are some additional tips for improving the speed and accuracy of the LLM:

* Use a pre-trained LLM model. This can save you a lot of time and effort in training the LLM.
* Use a good optimization algorithm. There are many different optimization algorithms available, and some are better suited for LLMs than others.
* Regularize the LLM. This can help to prevent the LLM from overfitting the training data.
* Use a validation set. The validation set is used to evaluate the performance of the LLM on unseen data. This can help you to avoid overfitting the training data.

I hope this helps!

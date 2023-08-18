### Step to run after setting up:
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


---
---

### References and Troublshooting of setting up

---

Sample query:

Enter a query: why was the NATO created?

---
Poetry install of Sentence_tranformers is incomplete
https://github.com/imartinez/privateGPT/issues/915


---
**How to install CUDA & cuDNN on Ubuntu 22.04**

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


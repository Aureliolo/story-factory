# Troubleshooting Guide

This guide covers common issues and their solutions for Story Factory.

## Table of Contents

- [Connection Issues](#connection-issues)
- [Performance Issues](#performance-issues)
- [Installation Issues](#installation-issues)
- [Model Issues](#model-issues)
- [UI Issues](#ui-issues)
- [Generation Issues](#generation-issues)
- [Platform-Specific Issues](#platform-specific-issues)

## Connection Issues

### Ollama Connection Refused

**Symptoms**:
- "Ollama offline" indicator in UI
- `ConnectionRefusedError` in logs
- Red status indicator in header

**Diagnosis**:
```bash
# Test Ollama connection
curl http://localhost:11434/api/tags
```

**Solutions**:

1. **Verify Ollama is running**:
   ```bash
   # Check if Ollama process is running
   # Windows: Task Manager → Services → Ollama
   # macOS: Activity Monitor → search "ollama"
   # Linux: systemctl status ollama
   ```

2. **Restart Ollama**:
   - **Windows**: Open Task Manager → Services → Right-click "Ollama" → Restart
   - **macOS**: 
     ```bash
     brew services restart ollama
     ```
   - **Linux**:
     ```bash
     sudo systemctl restart ollama
     ```

3. **Check firewall settings**:
   - Ensure port 11434 is not blocked
   - Allow Ollama through firewall/antivirus

4. **Verify settings.json**:
   ```json
   {
     "ollama_url": "http://localhost:11434"
   }
   ```
   Change if Ollama is running on a different host/port.

5. **Check Ollama logs**:
   - **Windows**: Event Viewer → Windows Logs → Application
   - **macOS**: `~/Library/Logs/Ollama/`
   - **Linux**: `sudo journalctl -u ollama -f`

### Network Timeout

**Symptoms**:
- Requests to Ollama timeout
- `TimeoutError` in logs

**Solutions**:

1. **Increase timeout in settings.json**:
   ```json
   {
     "ollama_timeout": 300,
     "ollama_generate_timeout": 180
   }
   ```

2. **Check system resources**: High CPU/memory usage can slow responses

3. **Restart both Ollama and Story Factory**

## Performance Issues

### Out of Memory (OOM)

**Symptoms**:
- System freezes or crashes
- "CUDA out of memory" errors
- Model fails to load

**Diagnosis**:
```bash
# Check VRAM usage
nvidia-smi

# Check system RAM
free -h  # Linux/macOS
```

**Solutions**:

1. **Use smaller models**:
   ```bash
   # Instead of 14B models, use 8B
   ollama pull huihui_ai/dolphin3-abliterated:8b
   ```

2. **Reduce context window** in settings.json:
   ```json
   {
     "context_size": 8192,  # Down from 32768
     "max_tokens": 2048     # Down from 8192
   }
   ```

3. **Use more aggressive quantization**:
   - Use Q4_K_M instead of Q8_0 or Q6_K
   - Smaller file size, slightly lower quality

4. **Disable per-agent models**:
   ```json
   {
     "use_per_agent_models": false,
     "default_model": "huihui_ai/dolphin3-abliterated:8b"
   }
   ```

5. **Close other GPU applications**:
   - Close games, video editors, other AI tools
   - Check Task Manager/Activity Monitor for GPU usage

6. **Enable sequential mode** (future feature):
   - Unload models between agent calls
   - Slower but uses less VRAM

### Slow Generation Speed

**Symptoms**:
- Story generation takes hours
- Models generate <5 tokens/second
- UI feels sluggish

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi

# Should show high GPU usage during generation
```

**Solutions**:

1. **Use faster models**:
   - 8B models are faster than 14B/30B
   - Dolphin is optimized for speed

2. **Reduce max tokens** in settings.json:
   ```json
   {
     "max_tokens": 2048,
     "previous_chapter_context_chars": 1000
   }
   ```

3. **Lower temperature** (slightly):
   - Higher temps = more computation
   - Try reducing writer temp from 0.9 to 0.8

4. **Enable checkpoint mode**:
   ```json
   {
     "interaction_mode": "checkpoint",
     "chapters_between_checkpoints": 3
   }
   ```
   Get intermediate results, validate progress

5. **Check GPU is being used**:
   ```bash
   # GPU should show high utilization during gen
   watch -n 1 nvidia-smi
   ```
   If GPU usage is low, Ollama may be using CPU

6. **Update Ollama**: Newer versions have performance improvements
   ```bash
   # Check version
   ollama --version
   
   # Update (varies by platform)
   # Windows: winget upgrade Ollama.Ollama
   # macOS: brew upgrade ollama
   # Linux: curl -fsSL https://ollama.com/install.sh | sh
   ```

### High CPU Usage

**Symptoms**:
- CPU at 100% during generation
- GPU underutilized
- System becomes unresponsive

**Possible Causes**:
- Ollama using CPU instead of GPU
- Model too large for VRAM, spilling to RAM
- Background processes competing

**Solutions**:

1. **Verify GPU support**:
   ```bash
   nvidia-smi  # Should show NVIDIA driver version
   ```

2. **Reinstall Ollama with GPU support**

3. **Use smaller models that fit in VRAM**

4. **Close background applications**

## Installation Issues

### pip Install Failures

**Symptoms**:
- `pip install -r requirements.txt` fails
- Missing dependencies errors
- Build errors for packages

**Solutions**:

1. **Update pip, setuptools, wheel**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Use virtual environment**:
   ```bash
   # Create venv
   python -m venv venv
   
   # Activate
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install build dependencies** (Linux):
   ```bash
   sudo apt-get install python3-dev build-essential
   ```

4. **Check Python version**:
   ```bash
   python --version  # Must be 3.13+
   ```

5. **Try installing packages individually**:
   ```bash
   pip install nicegui
   pip install ollama
   pip install pydantic
   # ... etc
   ```

### Python Version Issues

**Symptoms**:
- `SyntaxError` on modern Python syntax
- Import errors for type hints

**Solution**:
```bash
# Check version
python --version

# Must be 3.13 or higher
# Update Python or use pyenv/conda to manage versions
```

### Module Not Found Errors

**Symptoms**:
- `ModuleNotFoundError: No module named 'pydantic'`
- Even after pip install

**Solutions**:

1. **Check you're in the right environment**:
   ```bash
   which python  # Should point to your venv
   ```

2. **Reinstall requirements**:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

3. **Check PYTHONPATH**:
   ```bash
   # Should include your project directory
   echo $PYTHONPATH
   ```

## Model Issues

### Chinese Characters in Output

**Symptoms**:
- Story contains Chinese/CJK characters
- Mixed language output
- Usually with Qwen models

**Cause**: Qwen3 v1 abliterated has a known issue with Chinese "chain of thought" leaking.

**Solutions**:

1. **Switch to Dolphin** (recommended):
   ```bash
   ollama pull huihui_ai/dolphin3-abliterated:8b
   ```
   Update settings.json:
   ```json
   {
     "default_model": "huihui_ai/dolphin3-abliterated:8b"
   }
   ```

2. **Use Qwen3 v2** (if available):
   - Has layer-0 fix for this issue
   - Check Ollama library for updated versions

3. **Enable validator**:
   - Validator agent catches non-English output
   - May slow generation slightly

4. **Add explicit system prompt** (advanced):
   - Edit agent prompts to emphasize English only

### Model Not Found

**Symptoms**:
- "Model not found" error
- Agent fails to start

**Solutions**:

1. **Pull the model**:
   ```bash
   # Check available models
   ollama list
   
   # Pull missing model
   ollama pull <model-name>
   ```

2. **Check model name spelling** in settings.json

3. **Use "auto" for agent models**:
   ```json
   {
     "agent_models": {
       "writer": "auto",  # Auto-selects available model
       "architect": "auto"
     }
   }
   ```

### Model Download Fails

**Symptoms**:
- `ollama pull` hangs or fails
- Partial downloads

**Solutions**:

1. **Check internet connection**

2. **Free up disk space**: Models are large (5-50GB)

3. **Retry with timeout**:
   ```bash
   timeout 600 ollama pull <model>
   ```

4. **Use different mirror** (if available)

5. **Download manually** from HuggingFace:
   - Download GGUF file
   - Create Modelfile
   - Import to Ollama
   - See [docs/MODELS.md](docs/MODELS.md) for instructions

## UI Issues

### Blank Page / White Screen

**Symptoms**:
- Browser shows empty page
- No UI elements visible

**Solutions**:

1. **Check browser console** (F12):
   - Look for JavaScript errors
   - Note any failed network requests

2. **Force refresh**: Ctrl+Shift+R (clears cache)

3. **Try different browser**: Chrome, Firefox, Edge

4. **Check app is running**:
   ```bash
   curl http://localhost:7860
   # Should return HTML
   ```

5. **Check for port conflicts**:
   ```bash
   # Use different port
   python main.py --port 8080
   ```

6. **Disable browser extensions**:
   - Ad blockers can interfere
   - Try incognito/private mode

### UI Elements Not Responding

**Symptoms**:
- Buttons don't work
- Dropdowns won't open
- Forms don't submit

**Solutions**:

1. **Check browser console for errors**

2. **Refresh page**: F5 or Ctrl+R

3. **Clear browser cache and cookies**

4. **Update browser to latest version**

5. **Check JavaScript is enabled**

### Dark Mode Issues

**Symptoms**:
- Colors are wrong
- Text unreadable
- Theme doesn't persist

**Solutions**:

1. **Toggle dark mode**: Click sun/moon icon

2. **Check settings.json**:
   ```json
   {
     "dark_mode": true
   }
   ```

3. **Clear browser localStorage**: DevTools → Application → Local Storage

## Generation Issues

### Story Generation Stops

**Symptoms**:
- Generation freezes mid-chapter
- No error message
- Progress bar stuck

**Diagnosis**:
- Check logs: `logs/story_factory.log`
- Look for errors or timeouts

**Solutions**:

1. **Increase timeout** in settings.json:
   ```json
   {
     "ollama_timeout": 300,
     "llm_max_retries": 5
   }
   ```

2. **Check Ollama logs**:
   - Model may have crashed
   - Restart Ollama

3. **Reduce complexity**:
   - Shorter chapter targets
   - Simpler story concepts

4. **Try different model**:
   - Some models are more stable
   - Dolphin is very reliable

5. **Restart Story Factory**

### Poor Story Quality

**Symptoms**:
- Generic, repetitive prose
- Plot holes and inconsistencies
- Characters lack depth

**Solutions**:

1. **Use better models**:
   - Celeste V1.9 for creative writing
   - Larger models (14B+) for quality
   - See [docs/MODELS.md](docs/MODELS.md)

2. **Adjust temperatures**:
   ```json
   {
     "agent_temperatures": {
       "writer": 1.0,    # More creative
       "architect": 0.4  # More structured
     }
   }
   ```

3. **Provide detailed interview responses**:
   - More context = better output
   - Describe characters in detail
   - Specify tone and style preferences

4. **Use checkpoint mode**:
   - Review and provide feedback every N chapters
   - Guide the story direction

5. **Enable quality refinement** for world entities

### Repetitive Content

**Symptoms**:
- Same phrases repeated
- Characters act inconsistently
- Circular plot

**Solutions**:

1. **Increase temperature** (writer):
   ```json
   {"agent_temperatures": {"writer": 1.0}}
   ```

2. **Use different models for writer/editor**:
   - Variety in voice
   - Different creative approaches

3. **Enable continuity checker**:
   - Catches repetition
   - Forces revisions

4. **Provide feedback** at checkpoints:
   - Point out repetition
   - Request variety

## Platform-Specific Issues

### Windows

#### Ollama Service Won't Start

**Solutions**:
1. Open Services (services.msc)
2. Find "Ollama" service
3. Right-click → Start
4. If fails, check Event Viewer for errors

#### Firewall Blocking Connection

**Solutions**:
1. Windows Security → Firewall
2. Allow Ollama through firewall
3. Create inbound rule for port 11434

#### CUDA Not Detected

**Solutions**:
1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Verify with `nvidia-smi`
4. Restart after installation

### macOS

#### Ollama Not Running on Startup

**Solutions**:
```bash
# Enable auto-start
brew services start ollama

# Check status
brew services list
```

#### Permission Denied Errors

**Solutions**:
```bash
# Fix permissions
sudo chown -R $(whoami) /usr/local/bin/ollama
```

#### GPU Not Used (Apple Silicon)

**Note**: Apple Silicon (M1/M2/M3) uses Metal, not CUDA. Ollama should auto-detect. If not:

1. Update Ollama: `brew upgrade ollama`
2. Check Activity Monitor → GPU usage during generation

### Linux

#### Systemd Service Issues

**Solutions**:
```bash
# Check service status
sudo systemctl status ollama

# Restart service
sudo systemctl restart ollama

# View logs
sudo journalctl -u ollama -f

# Enable auto-start
sudo systemctl enable ollama
```

#### CUDA Library Not Found

**Solutions**:
```bash
# Install CUDA
sudo apt-get install nvidia-cuda-toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

#### Permission Denied for GPU

**Solutions**:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

## Getting More Help

If you're still experiencing issues:

1. **Check logs**: `logs/story_factory.log` for detailed error messages

2. **Enable debug logging**:
   ```bash
   python main.py --log-level DEBUG
   ```

3. **Search existing issues**: [GitHub Issues](https://github.com/Aureliolo/story-factory/issues)

4. **Create new issue** with:
   - Operating system and version
   - Python version (`python --version`)
   - Ollama version (`ollama --version`)
   - GPU info (`nvidia-smi`)
   - Full error message/traceback
   - Steps to reproduce
   - Relevant log excerpts

5. **Check documentation**:
   - [README.md](README.md) - Getting started
   - [docs/MODELS.md](docs/MODELS.md) - Model selection
   - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design

6. **Community resources**:
   - Ollama documentation: https://github.com/ollama/ollama
   - LocalLLaMA subreddit: r/LocalLLaMA
   - NiceGUI docs: https://nicegui.io

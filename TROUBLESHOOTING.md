# Troubleshooting Guide for Story Factory

This guide helps you resolve common issues with Story Factory.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Ollama Connection Problems](#ollama-connection-problems)
- [Model Issues](#model-issues)
- [Generation Problems](#generation-problems)
- [Performance Issues](#performance-issues)
- [Error Messages](#error-messages)

## Installation Issues

### Python Dependencies Won't Install

**Symptom**: `pip install -r requirements.txt` fails

**Solutions**:
1. Ensure you're using Python 3.10 or higher:
   ```bash
   python --version
   ```

2. Update pip:
   ```bash
   pip install --upgrade pip
   ```

3. Use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Gradio Port Already in Use

**Symptom**: "Address already in use" when starting the web UI

**Solution**: Use a different port:
```bash
python main.py --port 7861
```

## Ollama Connection Problems

### "Ollama not accessible" Error

**Symptom**: Cannot connect to Ollama server

**Solutions**:

1. **Check if Ollama is running**:
   ```bash
   ollama list
   ```
   If this fails, Ollama isn't running.

2. **Start Ollama** (it should auto-start, but if not):
   ```bash
   # Windows: Open Ollama from Start Menu
   # Linux/macOS: ollama serve
   ```

3. **Check Ollama URL**: Edit `settings.json` and ensure:
   ```json
   {
     "ollama_url": "http://localhost:11434"
   }
   ```

4. **Firewall issues**: Ensure port 11434 isn't blocked by your firewall.

### Ollama Running on Different Port/Host

**Solution**: Update `settings.json`:
```json
{
  "ollama_url": "http://your-host:your-port"
}
```

## Model Issues

### Model Not Found

**Symptom**: "Model 'xyz' not found" error

**Solutions**:

1. **Pull the model**:
   ```bash
   ollama pull model-name
   ```

2. **Check installed models**:
   ```bash
   ollama list
   ```

3. **Update settings**: In the web UI, go to Settings tab and:
   - Select an installed model from the dropdown
   - Or use "auto" to let the system choose

### Model Too Large for VRAM

**Symptom**: Generation is extremely slow or system crashes

**Solutions**:

1. **Use a smaller model**:
   - For 8GB VRAM: `huihui_ai/qwen3-abliterated:8b`
   - For 12GB VRAM: `huihui_ai/qwen3-abliterated:14b`
   - For 24GB+ VRAM: `huihui_ai/qwen3-abliterated:32b`

2. **Use quantized models**: Models with `:Q4_K_M` or similar tags use less VRAM

3. **Close other GPU applications**: Free up VRAM by closing games, browsers with hardware acceleration, etc.

4. **Check VRAM usage**:
   ```bash
   nvidia-smi
   ```

## Generation Problems

### Generation Gets Stuck

**Symptom**: Story generation stops responding

**Solutions**:

1. **Check logs**: Look in `logs/story_factory.log` for errors

2. **Increase timeout**: If using large models, they may take longer:
   - Check that your system isn't running out of memory
   - Monitor with `nvidia-smi` or `htop`

3. **Restart Ollama**:
   ```bash
   # Kill Ollama process and restart
   pkill ollama
   ollama serve
   ```

4. **Load saved story**: Stories auto-save after each chapter. Check `output/stories/` for recovery.

### Poor Quality Output

**Symptom**: Generated text is nonsensical or low quality

**Solutions**:

1. **Use a better model**: Larger models (32B, 70B) produce better quality

2. **Adjust temperature**: In Settings tab:
   - Lower temperature (0.6-0.7) for more focused writing
   - Higher temperature (0.9-1.0) for more creative writing

3. **Provide detailed requirements**: Give the Interviewer specific details about:
   - Genre and tone
   - Character personalities
   - Plot points
   - Setting details

4. **Increase context size**: In `settings.json`:
   ```json
   {
     "context_size": 32768,
     "max_tokens": 8192
   }
   ```

### Story Doesn't Match Requirements

**Symptom**: Generated story ignores user preferences

**Solutions**:

1. **Be more specific in interview**: Give concrete examples of what you want

2. **Use checkpoint mode**: Review every few chapters and provide feedback

3. **Edit the outline**: After the Architect phase, you can modify the outline before writing begins

## Performance Issues

### Slow Generation

**Symptom**: Each chapter takes a very long time

**Solutions**:

1. **Use faster models**:
   - 8B models are much faster than 70B
   - Check "speed" rating in model info

2. **Reduce max_tokens**: In `settings.json`:
   ```json
   {
     "max_tokens": 4096
   }
   ```

3. **Use per-agent models**: Let expensive models run only for the Writer:
   ```json
   {
     "use_per_agent_models": true,
     "agent_models": {
       "interviewer": "fast-8b-model",
       "architect": "fast-8b-model",
       "writer": "quality-32b-model",
       "editor": "fast-8b-model",
       "continuity": "fast-8b-model"
     }
   }
   ```

### High Memory Usage

**Symptom**: System runs out of RAM

**Solutions**:

1. **Use smaller context size**:
   ```json
   {
     "context_size": 16384
   }
   ```

2. **Reduce revision iterations**:
   ```json
   {
     "max_revision_iterations": 1
   }
   ```

3. **Close other applications**

## Error Messages

### "ValidationError: Story ID contains invalid characters"

**Cause**: Corrupted or manually edited story file

**Solution**: Don't manually edit story IDs. Create a new story instead.

### "LLMGenerationError: Failed to generate after 3 attempts"

**Cause**: Model crashed or Ollama connection interrupted

**Solutions**:
1. Check Ollama is running: `ollama list`
2. Check logs for details: `logs/story_factory.log`
3. Restart Ollama
4. Try a different model

### "ValueError: max_tokens cannot exceed context_size"

**Cause**: Invalid settings configuration

**Solution**: Edit `settings.json` to ensure `max_tokens <= context_size`:
```json
{
  "context_size": 32768,
  "max_tokens": 8192
}
```

### "FileNotFoundError: Story file not found"

**Cause**: Trying to load a non-existent story

**Solution**: 
1. Check `output/stories/` directory for available stories
2. Use `--list-stories` to see saved stories:
   ```bash
   python main.py --cli --list-stories
   ```

## Getting Help

If you're still experiencing issues:

1. **Check logs**: `logs/story_factory.log` contains detailed error information

2. **Enable debug logging**:
   ```bash
   python main.py --log-level DEBUG
   ```

3. **Report an issue**: Create a GitHub issue with:
   - Your Python version (`python --version`)
   - Your Ollama version (`ollama --version`)
   - Error message from logs
   - Steps to reproduce

4. **System information**:
   ```bash
   # GPU info
   nvidia-smi
   
   # Python packages
   pip list
   ```

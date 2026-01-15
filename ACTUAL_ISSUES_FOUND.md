# Actual Issues Found in Branch

## Real Problems

### 1. VRAM Unloading Doesn't Work (Bug)
**File:** `services/model_mode_service.py:223-234`

**Issue:** The `_unload_all_except()` method only tracks models in `self._loaded_models` but doesn't actually tell Ollama to unload anything. The comment says "Ollama will unload based on its own LRU cache" but that defeats the entire purpose of having a VRAM strategy.

**Impact:** Users selecting "Sequential" strategy expecting VRAM to be freed won't get that behavior.

**Fix:** Either:
1. Actually implement Ollama model unloading (if Ollama API supports it)
2. Document clearly that this is tracking-only and Ollama manages memory

### 2. Error Handling Missing in Analytics Page
**File:** `ui/pages/analytics.py`

**Issue:** Database queries in `_build_summary_section()` (lines 136-160) and `_build_model_section()` (lines 211-213) have no try/except. If database is locked or corrupted, UI crashes.

**Fix:** Wrap database calls and show user-friendly error message.

### 3. JSON Parsing Fragility in LLM Judge
**File:** `services/model_mode_service.py:366-377`

**Issue:** Uses basic regex `r"\{[^}]+\}"` which won't match nested JSON. Also imports `json` and `re` inside the try block which is inefficient.

**Actual problem:** This pattern won't match:
```json
{"prose_quality": 8.5, "metadata": {"notes": "good"}, "instruction_following": 7.0}
```

**Fix:** Use the existing `utils/json_parser.py` which handles this properly.

### 4. Mode Loading Can Crash on Bad Database Data
**File:** `services/model_mode_service.py:104-118`

**Issue:** If custom_modes table has invalid `vram_strategy` value, `VramStrategy(custom["vram_strategy"])` will raise `ValueError` and crash.

**Example:** Someone manually edits the database and sets vram_strategy to "fast"

**Fix:** Add validation with fallback:
```python
try:
    vram_strategy = VramStrategy(custom["vram_strategy"])
except ValueError:
    logger.warning(f"Invalid VRAM strategy in mode {custom['id']}, using ADAPTIVE")
    vram_strategy = VramStrategy.ADAPTIVE
```

### 5. CSV Export Uses Wrong Method
**File:** `ui/pages/analytics.py:322-389`

**Issue:** Calls `self._db.get_all_scores()` which doesn't exist in ModeDatabase!

Check the database class - there's no such method. This will crash when user clicks Export CSV.

**Fix:** Add the method to ModeDatabase or use `get_scores_for_project()` with filters.

## Things That Are Fine

✓ SQL injection - All queries use parameterized statements
✓ Database connections - Using `with` context managers correctly  
✓ Tests - 55 new tests, all passing
✓ Code formatting - Ruff clean
✓ Settings validation - Proper validation with clear error messages
✓ Pydantic models - Well-designed with appropriate validation

## Minor Improvements

### 6. Magic Number in LLM Judge
**File:** `services/model_mode_service.py:347`

`{content[:3000]}` - This 3000 should be a constant or configurable

### 7. Import Inside Function
**File:** `services/model_mode_service.py:366-367`

Move `import json` and `import re` to module top

## Testing the Branch

Ran basic smoke tests:
- ✅ All 228 tests pass
- ✅ Code lints and formats cleanly
- ✅ Analytics page instantiates without error
- ✅ Mode models validate correctly

## Priority Fixes

**Must fix before merge:**
1. Fix CSV export - add missing method or change call
2. Add error handling to Analytics UI
3. Fix JSON parsing in LLM judge

**Should fix:**
4. Mode loading validation
5. Document VRAM strategy limitations

**Nice to have:**
6. Extract magic numbers
7. Move imports to top

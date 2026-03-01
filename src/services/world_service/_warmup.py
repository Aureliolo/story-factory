"""Model warm-up for world build.

Pre-loads creator and judge models into Ollama VRAM before the build loop
to eliminate cold-start latency (investigation in #399 measured ~4.5s cold-start
penalty; warm-up itself costs ~2.5s, yielding ~2s net savings).
"""

import logging
import time
from typing import TYPE_CHECKING

from src.services.llm_client import get_ollama_client
from src.services.model_mode_service._vram import prepare_model
from src.utils.exceptions import VRAMAllocationError

if TYPE_CHECKING:
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)


def _warm_models(services: ServiceContainer) -> None:
    """Pre-load creator and judge models into Ollama VRAM before the build loop.

    Investigation (issue #399 script) measured a ~4.5s cold-start penalty for
    the first LLM call.  Sending a minimal ``chat(num_predict=1, num_ctx=512)``
    ping loads the model weights with minimal KV-cache allocation, reducing the
    first real call to warm-call latency (~2s net savings).

    Errors are logged but do not abort the build — the quality loop will
    handle any model loading failures during its own calls.
    """
    try:
        wq = services.world_quality
        creator_model = wq._get_creator_model()
        judge_model = wq._get_judge_model()
    except Exception:
        logger.warning(
            "Failed to resolve models for warm-up (non-fatal, build continues)",
            exc_info=True,
        )
        return

    # Preserve order (creator first, then judge if different) for deterministic logs
    models_to_warm = [creator_model]
    if judge_model != creator_model:
        models_to_warm.append(judge_model)

    for model in models_to_warm:
        t0 = time.perf_counter()
        try:
            # Run through prepare_model() first so eviction, residency checks,
            # and _loaded_models tracking all happen before the actual ping.
            prepare_model(services.mode, model)
            client = get_ollama_client(wq.settings, model_id=model)
            client.chat(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 512},
            )
            logger.info(
                "Warmed model '%s' into VRAM in %.2fs",
                model,
                time.perf_counter() - t0,
            )
        except VRAMAllocationError as e:
            logger.warning(
                "Model '%s' rejected by VRAM residency check (non-fatal, build continues): %s",
                model,
                e,
            )
        except Exception as e:
            logger.warning(
                "Failed to warm model '%s' (non-fatal, build continues): %s",
                model,
                e,
            )

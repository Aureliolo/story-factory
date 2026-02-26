"""Model warm-up for world build.

Pre-loads creator and judge models into Ollama VRAM before the build loop
to eliminate cold-start latency (~2s net savings per investigation #399).
"""

import logging
import time
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from src.services import ServiceContainer

logger = logging.getLogger(__name__)


def _warm_models(services: ServiceContainer) -> None:
    """Pre-load creator and judge models into Ollama VRAM before the build loop.

    Investigation (issue #399 script) measured a ~4.5s cold-start penalty for
    the first LLM call.  Sending a minimal ``chat(num_predict=1)`` ping loads
    the model weights and KV-cache, reducing the first real call to warm-call
    latency (~2s net savings).

    Errors are logged but do not abort the build — the quality loop will
    handle any model loading failures during its own calls.
    """
    wq = services.world_quality
    creator_model = wq._get_creator_model()
    judge_model = wq._get_judge_model()
    models_to_warm = list({creator_model, judge_model})  # deduplicate if same

    client = ollama.Client(
        host=wq.settings.ollama_url,
        timeout=wq.settings.ollama_generate_timeout,
    )

    for model in models_to_warm:
        t0 = time.perf_counter()
        try:
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
        except (ollama.ResponseError, Exception) as e:
            logger.warning(
                "Failed to warm model '%s' (non-fatal, build continues): %s",
                model,
                e,
            )

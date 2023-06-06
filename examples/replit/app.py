import json
import multiprocessing
from functools import partial
from threading import Lock
from typing import Dict, List, Optional, Union, Iterator, AsyncIterator

from main import ReplitModel, Completion, CompletionChunk

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model_file: str
    n_gpu_layers: int = 32
    n_batch: int = 512
    n_threads: int = max(multiprocessing.cpu_count() // 2, 1)


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = Field(
        default=16,
        ge=1,
        le=2048,
        description="The maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Adjust the randomness of the generated text.\n\n"
        + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
        + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
    )
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the results as they are generated. Useful for chatbots.",
    )
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
    )
    logprobs: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = Field(
        description="The model to use for generating completions."
    )
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


settings = Settings() # type: ignore
app = FastAPI()
model = ReplitModel.init_from_file(
    model_file=settings.model_file, n_gpu_layers=settings.n_gpu_layers
)
model_lock = Lock()


def get_model():
    with model_lock:
        yield model


# Used to support copilot.vim
@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    content = {"token": "1", "expires_at": 2600000000, "refresh_in": 900}
    return dict(status_code=200, content=content)


CreateCompletionResponse = create_model_from_typeddict(Completion)


# Used to support copilot.vim
@app.post(
    "/v1/engines/copilot-codex/completions",
    response_model=CreateCompletionResponse,
)
@app.post(
    "/v1/completions",
    response_model=CreateCompletionResponse,
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    model: ReplitModel = Depends(get_model),
):
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    exclude = {
        "n",
        "best_of",
        "logit_bias",
        "user",
    }
    kwargs = body.dict(exclude=exclude)
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        async def event_publisher(inner_send_chan: MemoryObjectSendStream[Dict[str, Union[str, bool]]]):
            async with inner_send_chan:
                try:
                    iterator: Iterator[CompletionChunk] = await run_in_threadpool(model.create_completion, **kwargs)  # type: ignore
                    async_iterator: AsyncIterator[CompletionChunk] = iterate_in_threadpool(iterator)
                    async for chunk in async_iterator:
                        await inner_send_chan.send(dict(data=json.dumps(chunk)))
                        if await request.is_disconnected():
                            raise anyio.get_cancelled_exc_class()()
                    await inner_send_chan.send(dict(data="[DONE]"))
                except anyio.get_cancelled_exc_class() as e:
                    print("disconnected")
                    with anyio.move_on_after(1, shield=True):
                        print(
                            f"Disconnected from client (via refresh/close) {request.client}"
                        )
                        await inner_send_chan.send(dict(closing=True))
                        raise e

        return EventSourceResponse(
            recv_chan, data_sender_callable=partial(event_publisher, send_chan)
        )
    else:
        completion: Completion = await run_in_threadpool(model.create_completion, **kwargs)  # type: ignore
        return completion